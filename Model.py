import torch
from torch import nn
import torch.nn.functional as F
from Params import args
import numpy as np
import random
from torch_scatter import scatter_sum, scatter_softmax
import math

init = nn.init.xavier_uniform_
uniformInit = nn.init.uniform

#用于知识图谱嵌入的表示学习
class Model(nn.Module):
	def __init__(self, handler):
		super(Model, self).__init__()

		#uEmbeds、eEmbeds 和 rEmbeds，分别代表用户嵌入、实体嵌入和关系嵌入
		#nn.Parameter用于将一个张量包装成模型的可学习参数，这意味着它在反向传播和优化过程中会被自动更新。
		#torch.empty(args.user, args.latdim)创建了一个大小为(args.user, args.latdim)的未初始化张量，args.user和args.latdim分别表示用户数量和潜在维度。
		self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))
		self.eEmbeds = nn.Parameter(init(torch.empty(args.entity_n, args.latdim)))
		self.rEmbeds = nn.Parameter(init(torch.empty(args.relation_num, args.latdim)))

		#nn.Sequential 用于定义一个包含多个神经网络层的序列，这些层按照顺序依次应用于输入数据。
		#*[GCNLayer() for i in range(args.gnn_layer)] 利用列表推导式生成包含 args.gnn_layer 个 GCNLayer 实例的列表，并通过 * 将其展开为参数传递给 nn.Sequential，这样就创建了包含多个 GCNLayer 实例的序列。
		# 76  每个 GCNLayer() 实例可能表示一个图卷积神经网络（GCN）层，用于在图数据上执行节点特征的聚合和更新操作。
		self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
		self.rgat = RGAT(args.latdim, args.layer_num_kg, args.mess_dropout_rate)#85 实现关系图注意力网络
		#                 嵌入大小         kg 层的数量          消息 dropout 率

		self.kg_dict = handler.kg_dict
		self.edge_index, self.edge_type = self.sampleEdgeFromDict(self.kg_dict, triplet_num=args.triplet_num)#57 从给定的知识图字典 kg_dict 中采样边
	
	def getEntityEmbeds(self):
		return self.eEmbeds#实体嵌入

	def getUserEmbeds(self):
		return self.uEmbeds#用户嵌入
				
	def forward(self, adj, mess_dropout=True, kg=None):
		if kg == None:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, [self.edge_index, self.edge_type], mess_dropout)
		else:
			hids_KG = self.rgat.forward(self.eEmbeds, self.rEmbeds, kg, mess_dropout)#return entity_res_emb
						
		embeds = torch.concat([self.uEmbeds, hids_KG[:args.item, :]], axis=0)#将用户嵌入和经过 rgat 处理后的嵌入信息合并
		embedsLst = [embeds]
		for gcn in self.gcnLayers:#循环迭代 gcnLayers 中的每一层，并将当前层的输出作为下一层的输入，不断传递信息并更新特征。
			embeds = gcn(adj, embedsLst[-1])
			embedsLst.append(embeds)
		embeds = sum(embedsLst)#将所有嵌入信息相加

		return embeds[:args.user], embeds[args.user:]#返回处理后的用户嵌入和物品嵌入数据。

	#从给定的知识图字典 kg_dict 中采样边
	def sampleEdgeFromDict(self, kg_dict, triplet_num=None):
		sampleEdges = []
		for h in kg_dict:
			t_list = kg_dict[h]
			if triplet_num != -1 and len(t_list) > triplet_num:
				sample_edges_i = random.sample(t_list, triplet_num)#在列表 t_list 中随机选择 triplet_num 个不重复的元素作为样本。
			else:
				sample_edges_i = t_list
			for r, t in sample_edges_i:
				sampleEdges.append([h, t, r])
		return self.getEdges(sampleEdges)# 69

	#对知识图边进行处理和转换为 PyTorch 张量
	def getEdges(self, kg_edges):
		graph_tensor = torch.tensor(kg_edges)
		index = graph_tensor[:, :-1]#获取张量中除最后一列以外的所有列
		type = graph_tensor[:, -1]#获取张量的最后一列，这部分被表示为类型或标签
		return index.t().long().cuda(), type.long().cuda()

#执行图卷积网络中的一层操作
class GCNLayer(nn.Module):
	def __init__(self):
		super(GCNLayer, self).__init__()

	#该层接收邻接矩阵(adj)和节点特征表示(embeds)，然后通过计算稀疏矩阵乘法，返回更新后的节点表示
	def forward(self, adj, embeds):
		return torch.spmm(adj, embeds)

#实现关系图注意力网络
class RGAT(nn.Module):
	##                 嵌入大小   kg 层的数量   消息 dropout 率
	def __init__(self, latdim, n_hops, mess_dropout_rate=0.4):
		super(RGAT, self).__init__()
		self.mess_dropout_rate = mess_dropout_rate
		#nn.Parameter 用于将一个张量包装为可学习的参数，这意味着它将会在网络的训练过程中进行更新。
		# torch.empty(size=(2*latdim, latdim)) 创建了一个大小为 (2*latdim, latdim) 的未初始化张量作为参数 W 的初始数值。
		# nn.init.calculate_gain('relu') 用于计算初始化时的增益值，这样可以帮助加速收敛和稳定训练过程，尤其是在使用 ReLU 激活函数时。
		self.W = nn.Parameter(init(torch.empty(size=(2*latdim, latdim)), gain=nn.init.calculate_gain('relu')))

		self.leakyrelu = nn.LeakyReLU(0.2)# LeakyReLU 激活函数的实例，其中 0.2 是负半轴上的斜率值。
		self.n_hops = n_hops
		self.dropout = nn.Dropout(p=mess_dropout_rate)#创建了一个 Dropout 层的实例,mess_dropout_rate 参数用于指定丢弃概率 p，即在训练过程中随机地“丢弃”输入单元以减少过拟合。

	#消息聚合操作，根据节点和关系嵌入以及知识图（kg）的边索引，计算了节点之间的关系注意力权重，进行了消息传递和聚合操作
	def agg(self, entity_emb, relation_emb, kg):
		edge_index, edge_type = kg
		head, tail = edge_index
		a_input = torch.cat([entity_emb[head], entity_emb[tail]], dim=-1)#将头部和尾部节点的嵌入拼接在一起
		e_input = torch.multiply(torch.mm(a_input, self.W), relation_emb[edge_type]).sum(-1)#通过矩阵乘法、点乘和汇总得到节点之间的注意力系数
		e = self.leakyrelu(e_input)
		e = scatter_softmax(e, head, dim=0, dim_size=entity_emb.shape[0])
		agg_emb = entity_emb[tail] * e.view(-1, 1)#对经过关系注意力权重加权后的尾部节点嵌入 entity_emb[tail] 与归一化后的注意力系数 e（reshape 到列向量形状）进行点乘，得到每个节点对应的加权后的嵌入信息。
		agg_emb = scatter_sum(agg_emb, head, dim=0, dim_size=entity_emb.shape[0])#对加权后的嵌入信息 agg_emb 进行求和，确保每个节点聚合到对应头部节点处。
		agg_emb = agg_emb + entity_emb#将聚合后的节点嵌入信息 agg_emb 与原始节点嵌入信息相加，得到最终更新后的节点表示信息。
		return agg_emb

	#前向传播方法，用于执行多个消息传递和聚合的循环
	def forward(self, entity_emb, relation_emb, kg, mess_dropout=True):
		            #self.eEmbeds, self.rEmbeds, kg, mess_dropout
		entity_res_emb = entity_emb
		#调用 agg 方法进行消息聚合，根据需求进行消息丢弃处理，对节点嵌入进行规范化，并更新节点表示。
		for _ in range(self.n_hops):#消息传递次数 n_hops
			entity_emb = self.agg(entity_emb, relation_emb, kg)
			if mess_dropout:#如果 mess_dropout 为真，则将节点嵌入信息通过 dropout 方法进行消息丢弃处理。
				entity_emb = self.dropout(entity_emb)
			entity_emb = F.normalize(entity_emb)#对处理后的节点嵌入进行规范化（Normalization），确保节点表示的稳定性和可比性。

			#entity_res_emb = args.res_lambda * entity_res_emb + entity_emb
		#利用参数 res_lambda 来调整之前节点表示信息 entity_res_emb 和当前更新后节点表示信息 entity_emb 之间的比例关系，并将更新后的节点表示信息保存到 entity_res_emb 中。
		return entity_res_emb

#神经网络模型，旨在对输入数据进行噪声去除（denoising）操作
class Denoise(nn.Module):
	#初始化神经网络结构
	def __init__(self, in_dims, out_dims, emb_size, norm=False, dropout=0.5):
		super(Denoise, self).__init__()
		self.in_dims = in_dims
		self.out_dims = out_dims
		self.time_emb_dim = emb_size
		self.norm = norm

		self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)#初始化了一个线性层。线性层的输入维度为 self.time_emb_dim，输出维度也为 self.time_emb_dim

		in_dims_temp = [self.in_dims[0] + self.time_emb_dim + args.entity_n] + self.in_dims[1:]
		#in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]

		out_dims_temp = self.out_dims

		#用于存储多个线性层。这些线性层将根据给定的输入维度列表 in_dims_temp 和输出维度列表 out_dims_temp 实例化，并存储在对应的 ModuleList 中
		self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
		self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) for d_in, d_out in zip(out_dims_temp[:-1], out_dims_temp[1:])])

		self.drop = nn.Dropout(dropout)
		self.init_weights()#149

	#初始化神经网络各层的权重和偏置。根据输入输出层权重的维度计算标准差并进行正态分布初始化。
	def init_weights(self):
		for layer in self.in_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)
		
		for layer in self.out_layers:
			size = layer.weight.size()
			std = np.sqrt(2.0 / (size[0] + size[1]))
			layer.weight.data.normal_(0.0, std)
			layer.bias.data.normal_(0.0, 0.001)

		size = self.emb_layer.weight.size()
		std = np.sqrt(2.0 / (size[0] + size[1]))
		self.emb_layer.weight.data.normal_(0.0, std)
		self.emb_layer.bias.data.normal_(0.0, 0.001)

	#执行前向传播。根据输入的数据和时间步信息，计算时间嵌入，拼接嵌入和输入数据，经过多层神经网络处理后返回输出结果
	def forward(self, x, timesteps, item_entity, mess_dropout=True):
		#计算频率和时间步信息，生成时间嵌入 time_emb
		freqs = torch.exp(-math.log(10000) * torch.arange(start=0, end=self.time_emb_dim//2, dtype=torch.float32) / (self.time_emb_dim//2)).cuda()
		temp = timesteps[:, None].float() * freqs[None]
		time_emb = torch.cat([torch.cos(temp), torch.sin(temp)], dim=-1)
		if self.time_emb_dim % 2:
			time_emb = torch.cat([time_emb, torch.zeros_like(time_emb[:, :1])], dim=-1)
		emb = self.emb_layer(time_emb)#对输入的时间嵌入 time_emb 进行线性映射处理，得到 emb
		if self.norm:
			x = F.normalize(x)#对输入 x 进行归一化处理
			item_entity = F.normalize(item_entity)
		if mess_dropout:
			x = self.drop(x)#对 x 进行丢弃操作

		#h = torch.cat([x, emb], dim=-1)
		h = torch.cat([x, emb, item_entity], dim=-1)#将处理后的 x 与 emb 进行拼接

		for i, layer in enumerate(self.in_layers):
			h = layer(h)
			h = torch.tanh(h)
		for i, layer in enumerate(self.out_layers):
			h = layer(h)
			if i != len(self.out_layers) - 1:
				h = torch.tanh(h)

		return h

#实现高斯扩散
class GaussianDiffusion(nn.Module):
	#初始化高斯扩散模型，设置初始噪声的尺度参数、噪声的最小值和最大值,GaussianDiffusion 模型的步数(default=5),是否固定 beta 参数
	def __init__(self, noise_scale, noise_min, noise_max, steps, beta_fixed=True):
		super(GaussianDiffusion, self).__init__()

		self.noise_scale = noise_scale
		self.noise_min = noise_min
		self.noise_max = noise_max
		self.steps = steps

		if noise_scale != 0:
			self.betas = torch.tensor(self.get_betas(), dtype=torch.float64).cuda()#208
			if beta_fixed:
				self.betas[0] = 0.0001

			self.calculate_for_diffusion()#220

	#计算并返回一组 beta 值，用于高斯扩散过程中 alpha 和 beta 之间的转换
	def get_betas(self):
		start = self.noise_scale * self.noise_min
		end = self.noise_scale * self.noise_max
		#起始和终止点之间生成具有固定步长的一维数组，用于表示噪声的变化范围
		variance = np.linspace(start, end, self.steps, dtype=np.float64)
		alpha_bar = 1 - variance
		betas = []
		betas.append(1 - alpha_bar[0])
		for i in range(1, self.steps):
			betas.append(min(1 - alpha_bar[i] / alpha_bar[i-1], 0.999))
		return np.array(betas)

	#根据计算得到的 beta 值，计算扩散过程中需要用到的其他参数和变量。
	def calculate_for_diffusion(self):
		alphas = 1.0 - self.betas
		self.alphas_cumprod = torch.cumprod(alphas, axis=0).cuda()#沿着列的方向，对每列中的元素进行累积乘积计算
		# 将单个元素 1.0 与 alphas_cumprod 中除最后一个元素外的部分进行拼接
		self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]).cuda(), self.alphas_cumprod[:-1]]).cuda()
		#将 alphas_cumprod 中从第二个元素开始的部分与一个包含单个元素 0.0 的张量进行拼接。
		self.alphas_cumprod_next = torch.cat([self.alphas_cumprod[1:], torch.tensor([0.0]).cuda()]).cuda()

		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)#计算给定张量 self.alphas_cumprod 中每个元素的平方根
		self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)#计算给定张量 self.alphas_cumprod 中每个元素与 1.0 的差值的平方根。
		self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)#计算给定张量 self.alphas_cumprod 中每个元素与 1.0 的差值的自然对数。
		self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)#计算给定张量 self.alphas_cumprod 中每个元素的倒数的平方根。
		self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)#计算给定张量 self.alphas_cumprod 中每个元素的倒数减去 1 后的结果的平方根。

		#计算后验方差 posterior_variance
		self.posterior_variance = (
			self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
		)
		self.posterior_log_variance_clipped = torch.log(torch.cat([self.posterior_variance[1].unsqueeze(0), self.posterior_variance[1:]]))
		#后验均值的系数
		self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
		self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - self.alphas_cumprod))

	#实现扩散过程中的样本抽取
	def p_sample(self, model, x_start, steps, batch_index, itmEmbeds):
		if steps == 0:
			x_t = x_start
		else:
			t = torch.tensor([steps-1] * x_start.shape[0]).cuda()
			x_t = self.q_sample(x_start, t)
		
		indices = list(range(self.steps))[::-1]#创建一个倒序的步骤索引列表 indices
		item_entity = itmEmbeds[batch_index] @ itmEmbeds.T

		for i in indices:
			t = torch.tensor([i] * x_t.shape[0]).cuda()#创建了一个张量，其中每个元素都是 i，并且这个张量的长度与 x_t 张量的第一个维度长度相同
			model_mean, model_log_variance = self.p_mean_variance(model, x_t, t, item_entity)#277 实现扩散过程中的均值方差计算
			x_t = model_mean
		return x_t
			
	def q_sample(self, x_start, t, noise=None):
		if noise is None:
			noise = torch.randn_like(x_start)
		return self._extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + self._extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise


	def _extract_into_tensor(self, arr, timesteps, broadcast_shape):
		arr = arr.cuda()
		res = arr[timesteps].float()
		while len(res.shape) < len(broadcast_shape):
			res = res[..., None]
		return res.expand(broadcast_shape)

	# 实现扩散过程中的均值方差计算
	def p_mean_variance(self, model, x, t, item_entity):

		model_output = model(x, t, item_entity, False)#对输入数据 x_t 进行噪声去除的处理，调用了 denoise_model 实例的前向传播方法

		model_variance = self.posterior_variance#后验方差
		model_log_variance = self.posterior_log_variance_clipped#后验对数方差

		model_variance = self._extract_into_tensor(model_variance, t, x.shape)
		model_log_variance = self._extract_into_tensor(model_log_variance, t, x.shape)

		model_mean = (self._extract_into_tensor(self.posterior_mean_coef1, t, x.shape) * model_output + self._extract_into_tensor(self.posterior_mean_coef2, t, x.shape) * x)
		
		return model_mean, model_log_variance

	#计算训练损失，包括扩散损失和其他自定义损失函数的计算过程。
	def training_losses(self, model, x_start, ui_matrix, userEmbeds, itmEmbeds, batch_index):
		            #sdenoise_model, batch_item, ui_matrix, uEmbeds, iEmbeds, batch_index
		batch_size = x_start.size(0)#用于获取 PyTorch 张量（tensor） x_start 的第一个维度的大小（即张量的行数或第一个维度的长度）。
		ts = torch.randint(0, self.steps, (batch_size,)).long().cuda()#0 是生成随机整数的最小值。self.steps 是生成随机整数的最大值。(batch_size,) 是生成的随机整数张量的形状，
		# 这里是一个一维张量，包含 batch_size 个随机整数
		noise = torch.randn_like(x_start)#创建一个形状与 x_start 相同，且值服从标准正态分布的新张量
		if self.noise_scale != 0:
			x_t = self.q_sample(x_start, ts, noise)
		else:
			x_t = x_start

		item_entity = itmEmbeds[batch_index] @ itmEmbeds.T

		#来自知识图的预测关系概率
		model_output = model(x_t, ts, item_entity)#对输入数据 x_t 进行噪声去除的处理，调用了 denoise_model 实例的前向传播方法

		#均方误差（MSE）
		mse = self.mean_flat((x_start - model_output) ** 2)#318

		weight = self.SNR(ts - 1) - self.SNR(ts)#324 计算了两个时刻 ts - 1 和 ts 下的信噪比（SNR）之差
		weight = torch.where((ts == 0), 1.0, weight)#如果 ts 的值为 0，则将 weight 设置为1.0

		#ELBO 损失    信噪比差异调整后的损失项
		diff_loss = weight * mse#weight调节了不同时间步之间损失的权重

		item_user_matrix = torch.spmm(ui_matrix, model_output[:, :args.item].t()).t()#计算用户-物品矩阵（ui_matrix）与model_output的前 args.item 列之间的乘积
		itmEmbeds_kg = torch.mm(item_user_matrix, userEmbeds)#对用户-物品关系矩阵和用户Embeddings进行矩阵相乘，得到的结果将包含了物品与用户Embeddings之间的关系
		# 协作知识图卷积的损失CKGC
		ukgc_loss = self.mean_flat((itmEmbeds_kg - itmEmbeds[batch_index]) ** 2)

		return diff_loss, ukgc_loss#ELBO损失   CKGC损失

	#计算输入张量的均值
	def mean_flat(self, tensor):
		#使用 mean 函数计算输入张量在除去第一个维度（通常是 batch 维度）以外的所有维度上的均值。
		# 通过指定 dim=list(range(1, len(tensor.shape)))，在所有维度上计算均值，除去第一个维度（通常是 batch 维度）
		return tensor.mean(dim=list(range(1, len(tensor.shape))))

	#计算信噪比
	def SNR(self, t):
		self.alphas_cumprod = self.alphas_cumprod.cuda()#将存储在 self.alphas_cumprod 中的累积因子转移到 GPU 上进行处理
		return self.alphas_cumprod[t] / (1 - self.alphas_cumprod[t])#该时间步t下的信噪比值