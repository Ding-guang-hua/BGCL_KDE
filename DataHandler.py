import pickle
import numpy as np
from scipy.sparse import coo_matrix,csr_matrix
from Params import args
import scipy.sparse as sp
import torch
import torch.utils.data as data
import torch.utils.data as dataloader
from collections import defaultdict
from tqdm import tqdm
import random
from Utils.TimeLogger import log

class DataHandler:
	def __init__(self):
		if args.data == 'movie':
			predir = './Datasets/movie-lens/'
		elif args.data == 'book':
			predir = './Datasets/amazon-book/'
		elif args.data == 'music':
			predir = './Datasets/music-small/'
		elif args.data == 'luo':
			predir = './Datasets/luo/'
		self.predir = predir
		self.trnfile = predir + 'trnMat.pkl'
		self.tstfile = predir + 'tstMat.pkl'
		self.kgfile = predir + 'kg.txt'
		self.S_file = predir + 'Similarity_' + args.similarity + '.pkl'

	#加载一个文件，并将其内容转换为稀疏矩阵
	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:#打开给定的文件,文件名由变量 filename 指定，并使用 'rb' 模式(二进制读取)进行打开,fs 是一个变量名，用于引用打开的文件对象
			ret = (pickle.load(fs) != 0).astype(np.float32)#加载数据(将二进制数据转换回原始的 Python 对象)并检查非零元素，将其转换为 np.float32 类型的值
		if type(ret) != coo_matrix:#检查 ret 的类型是否为 coo_matrix
			ret = sp.coo_matrix(ret)#如果 ret 不是 coo_matrix 类型，则将其转换为 COO 格式的稀疏矩阵

		return ret

	#读取一个文件，进行一系列处理
	def readTriplets(self, file_name):
		can_triplets_np = np.loadtxt(file_name, dtype=np.int32)#从 file_name 文件加载数据为 NumPy 数组并指定为 np.int32 类型
		can_triplets_np = np.unique(can_triplets_np, axis=0)#对加载的数据进行去重处理,保留唯一的三元组数据

		inv_triplets_np = can_triplets_np.copy()
		inv_triplets_np[:, 0] = can_triplets_np[:, 2]
		inv_triplets_np[:, 2] = can_triplets_np[:, 0]
		inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
		triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)#将两个 NumPy 数组连接起来形成 triplets

		n_relations = max(triplets[:, 1]) + 1

		args.relation_num = n_relations

		args.entity_n = max(max(triplets[:, 0]), max(triplets[:, 1])) + 1

		return triplets
	
	#用来构建知识图谱
	def buildGraphs(self, triplets):
		kg_dict = defaultdict(list)#创建一个名为 kg_dict 的默认字典，用于存储知识图数据的边和相应的信息
        # h, t, r
		kg_edges = list()

		print("Begin to load knowledge graph triples ...")

		kg_counter_dict = {}
		#遍历三元组列表 triplets，逐个处理其中的 (h_id, r_id, t_id)
		for h_id, r_id, t_id in tqdm(triplets, ascii=True):#tqdm 函数用于在循环中显示进度条，让你能够实时查看循环的执行进度。ascii=True 参数指定使用 ASCII 字符显示进度条
			if h_id not in kg_counter_dict.keys():
				kg_counter_dict[h_id] = set()#如果 h_id 不在 kg_counter_dict 的键中，则将其初始化为一个空集合
			if t_id not in kg_counter_dict[h_id]:
				kg_counter_dict[h_id].add(t_id)
			else:
				continue
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))

		return kg_edges, kg_dict
	
	#构建知识图谱的邻接矩阵
	def buildKGMatrix(self, kg_edges):
		edge_list = []
		for h_id, t_id, r_id in kg_edges:
			edge_list.append((h_id, t_id))
		edge_list = np.array(edge_list)#将 edge_list 转换为 NumPy 数组，
		# 使用 SciPy 库中的 csr_matrix 函数创建一个压缩稀疏行（CSR）格式的稀疏矩阵。
		#创建一个与 edge_list 第一列形状相同的元素值全部为 1 的 NumPy 数组；这个数组用来表示稀疏矩阵中的非零元素
		# (edge_list[:,0], edge_list[:,1]): 这是一个元组，用来提供稀疏矩阵非零元素的位置索引。edge_list 中第一列的值用作行索引，第二列的值用作列索引，表示非零元素所在的位置。
		kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(args.entity_n, args.entity_n))

		return kgMatrix

	#对输入的稀疏矩阵进行归一化处理
	def normalizeAdj(self, mat, S_values):
		degree = np.array(mat.sum(axis=-1)) #计算输入矩阵 mat 沿着最后一个维度（即每一行的方向）的和，得到节点的度信息，构建一个数组 degree
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])#计算度的倒数平方，并将结果重塑为一维数组 dInvSqrt
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0#无穷值（如果有）替换为 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)#使用 sp.diags 创建一个对角矩阵 dInvSqrtMat，用于存储度的倒数平方
		mat = mat + S_values
		#print("Effectiveness of Attention-aware Matrix")
		return dInvSqrtMat.transpose().dot(mat).dot(dInvSqrtMat).tocoo()
		#将输入矩阵 mat 与 dInvSqrtMat 相乘，然后转置，再与 dInvSqrtMat 相乘，得到归一化后的结果。最后将结果转换为 COO (Coordinate list) 格式并返回


	# Perform steps random walk from start_node
	def random_walk(self,adj_matrix, start_node, steps):
		current_node = start_node
		walk_sequence = []
		previous_node = start_node
		restart = 0

		for _ in range(steps):
			neighbors = adj_matrix[current_node].indices  # Get neighboring nodes
			#print(current_node,neighbors,type(neighbors))
			if restart > random.random():
				current_node = previous_node
				walk_sequence.pop()
				restart = 0
				_ -= 1
				continue
			else:
				if len(neighbors) > 0:
					previous_node = current_node
					current_node = np.random.choice(neighbors)  # Move to a random neighbor
					walk_sequence.append(current_node)
				else:
					break
			restart += 0.2

		return walk_sequence

	def k_adjacency_matrix(self, adjacency_matrix, k):
		adjacency_matrix_power = {}
		kadj_set = [[] for _ in range(adjacency_matrix.shape[0])]
		adjacency_matrix_power[1] = adjacency_matrix
		for i in range(2, k + 1):
			adjacency_matrix_power[i] = adjacency_matrix_power[i - 1].dot(adjacency_matrix)
			print(i,"阶邻居节点")
		for i in range(adjacency_matrix.shape[0]):
			for j in range(1, k + 1):
				neighbors_indices = set(adjacency_matrix_power[j].getrow(i).indices) - {i}
				kadj_set[i].append(neighbors_indices)

		return kadj_set
	#计算相似度矩阵
	def calculate_S_from_adjacency_matrix(self, adj_matrix, L, M, k, epsilon):
		print("开始构建相似度矩阵")
		S_values = np.zeros(adj_matrix.shape)
		kadj_set = self.k_adjacency_matrix(adj_matrix, k)

		for u in range(adj_matrix.shape[0]):
			adj_matrix_u = adj_matrix.getrow(u).toarray()[0]
			for v in range(adj_matrix.shape[1]):
				if v > u:
					break
				if adj_matrix[u, v] == 0:  # 如果节点对(u, v)不在邻接矩阵中则跳过
					continue
				adj_matrix_v = adj_matrix.getcol(v).toarray().T[0]
				# print("-------------------------")
				common_neighbors = 0
				u_m_l = []
				v_m_l = []
				for _ in range(M):
					start_node_u = u  # Start node for random walk in u
					start_node_v = v  # Start node for random walk in v
					u_walk_sequence = self.random_walk(adj_matrix, start_node_u, L)  # Perform L-step random walk from u
					v_walk_sequence = self.random_walk(adj_matrix, start_node_v, L)  # Perform L-step random walk from v

					u_m_l.append(u_walk_sequence)
					v_m_l.append(v_walk_sequence)
				# 高阶邻居集合
				u_flat = set([node for seq in u_m_l for node in seq])
				v_flat = set([node for seq in v_m_l for node in seq])
				print("-----------------",len(u_flat),len(v_flat))

				# k阶邻居集合
				uSet = set()
				vSet = set()
				for i in range(1, k + 1):
					u_kSet = kadj_set[u][i-1]
					v_kSet = kadj_set[v][i-1]
					uSet = uSet.union(u_kSet)
					vSet = vSet.union(v_kSet)
				a = uSet.intersection(u_flat)
				b = vSet.intersection(v_flat)
				common_neighbors = len(a.intersection(b))
				print(common_neighbors)
				non_zero_u = np.count_nonzero(adj_matrix_u)
				non_zero_v = np.count_nonzero(adj_matrix_v)

				if non_zero_u > 0 and non_zero_v > 0:
					S_uv = epsilon * common_neighbors / (non_zero_u + non_zero_v)
					S_values[u][v] = S_uv
					S_values[v][u] = S_uv
				print('相似度 %d/%d' % (u, v), S_uv)
				#log('相似度 %d/%d: %d  %d' % (u, v, common_neighbors,S_uv), save=False, oneline=True)
		S_values = csr_matrix(S_values)
		with open(self.predir+'Similarity.pkl', 'wb') as file:
			pickle.dump(S_values, file)
		print(type(S_values))
		return S_values

	#将输入的稀疏矩阵转换为 PyTorch 的稀疏张量
	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))#创建两个空的稀疏矩阵 a 和 b，大小分别为 (args.user, args.user) 和 (args.item, args.item)
		b = sp.csr_matrix((args.item, args.item))
		#sp.vstack: 这个函数用于在垂直方向（竖直方向）堆叠稀疏矩阵，将它们垂直地拼接在一起。sp.hstack: 这个函数用于在水平方向（横向方向）堆叠稀疏矩阵，将它们水平地拼接在一起
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])#将这两个稀疏矩阵与输入的稀疏矩阵 mat 垂直和水平拼接，构成新的稀疏矩阵 mat
		mat = (mat != 0) * 1.0#将稀疏矩阵中不为零的元素置为1
		#L步M次带重启随机游走，k阶邻居节点  计算相似度矩阵
		#self.calculate_S_from_adjacency_matrix(mat, args.L, args.M, args.k, args.epsilon)

		with open(self.S_file, 'rb') as file:
			 S_values= pickle.load(file)
		print("S_values-----",self.S_file,args.epsilon)

		S_values = args.epsilon * S_values
		mat = (mat + sp.eye(mat.shape[0])) * 1.0#将对角线上的元素加1   邻接矩阵

		mat = self.normalizeAdj(mat, S_values)#96对输入的稀疏矩阵进行归一化处理


		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		#创建 PyTorch 张量，该张量包含了 mat 的行索引和列索引。这里使用 np.vstack([mat.row, mat.col]) 将 mat 的行和列索引竖直堆叠在一起作为索引，然后将其转换为 int64 类型。
		vals = torch.from_numpy(mat.data.astype(np.float32))#创建 PyTorch 张量，包含了 mat 的数据，这里将数据类型转换为 float32
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	#构建一个表示关系的字典
	def RelationDictBuild(self):
		relation_dict = {}
		#对于输入的 kg_dict 中的每个头部实体 head，创建一个空字典，并将这个空字典作为值与头部实体 head 相关联，形成嵌套字典结构
		for head in self.kg_dict:
			relation_dict[head] = {}
			for (relation, tail) in self.kg_dict[head]:
				relation_dict[head][tail] = relation
		return relation_dict

	#将输入的稀疏矩阵转换为 PyTorch 的稀疏张量并将其移动到 GPU 上
	def buildUIMatrix(self, mat):
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))#将稀疏矩阵的行索引和列索引堆叠在一起，形成索引数组 idxs，并将其转换为 PyTorch 的张量。
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()#使用 torch.sparse.FloatTensor 创建 PyTorch 的稀疏张量，参数为索引 idxs、值 vals 和形状 shape

	#加载和准备数据以及构建各种数据结构用于模型训练
	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)#35加载一个文件，并将其内容转换为稀疏矩阵
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat

		args.user, args.item = trnMat.shape#从训练数据矩阵的形状中获取用户数和物品数
		#print("-----------------",args.user,args.item)
		#根据trnmat计算s(u,i)-------------------------------------------------------------------
		self.torchBiAdj = self.makeTorchAdj(trnMat)#105将输入的稀疏矩阵转换为 PyTorch 的稀疏张量

		self.ui_matrix = self.buildUIMatrix(trnMat)#132使用 buildUIMatrix 方法从 trnMat 构建 PyTorch 稀疏矩阵 ui_matrix

		trnData = TrnData(trnMat)#169用训练数据矩阵创建训练数据集对象 trnData
		#dataloader.DataLoader: 这是一个 DataLoader 对象的构造函数，用于加载数据并生成一个数据迭代器。
		# trnData: 这是包含训练数据的数据集对象，它将在数据加载器中使用。
		# batch_size=args.batch: 这指定了每个小批次中的样本数量，该数量通常由参数 args.batch 指定。
		# shuffle=True: 这表示在每个 epoch 开始之前是否随机打乱数据。这有助于训练模型时减少过拟合。
		#num_workers=0: 这指定了用于数据加载的子进程数。设置为0意味着数据将在主进程中加载，而不是使用多个子进程来加速数据加载。
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)#199
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		kg_triplets = self.readTriplets(self.kgfile)#35
		# 从文件中读取知识图谱三元组
		self.kg_edges, self.kg_dict = self.buildGraphs(kg_triplets)#63构建知识图谱

		self.kg_matrix = self.buildKGMatrix(self.kg_edges)#85构建知识图谱的邻接矩阵
		print("kg shape: ", self.kg_matrix.shape)
		print("number of edges in KG: ", len(self.kg_edges))
		
		self.diffusionData = DiffusionData(torch.FloatTensor(self.kg_matrix.A))#202 根据转换为 PyTorch 张量的知识图谱矩阵创建扩散数据
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

		self.relation_dict = self.RelationDictBuild()#112

#处理训练数据集
class TrnData(data.Dataset):
	#接受一个稀疏矩阵 coomat 作为参数，初始化对象的行、列、稀疏 DOK（Dictionary of Keys）矩阵以及负样本数组
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()#将稀疏矩阵转换为 DOK 格式
		self.negs = np.zeros(len(self.rows)).astype(np.int32)#创建一个与行数相同长度的零向量，并将其转换为 int32 类型

	def negSampling(self):#对每个正样本生成一个负样本
		for i in range(len(self.rows)):
			#随机选择一个负样本，直到负样本与对应的用户不在稀疏矩阵中。负样本的索引存储在 self.negs 数组中
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(args.item)
				if (u, iNeg) not in self.dokmat:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.rows)

	def __getitem__(self, idx):
		return self.rows[idx], self.cols[idx], self.negs[idx]

#处理测试数据集
class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):#初始化测试数据集类的实例
		self.csrmat = (trnMat.tocsr() != 0) * 1.0#将稀疏矩阵 trnMat 转换为 CSR 格式，并将非零元素转换为1.0的二元稀疏矩阵。
		self.dokmat = coomat.todok()  # 将稀疏矩阵转换为 DOK 格式
		self.dokmat_trn = trnMat.todok()
		self.rows = coomat.row
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
		self.item = np.max(coomat.col)
		self.negSampling()

		tstLocs = [None] * coomat.shape[0]#创建了一个长度为 coomat.shape[0] 的列表 tstLocs，其中每个元素都初始化为 None
		tstNegs = [None] * coomat.shape[0]
		tstUsrs = set()
		for i in range(len(coomat.data)):
			row = coomat.row[i]
			col = coomat.col[i]
			neg = self.negs[i]
			if tstLocs[row] is None:
				tstLocs[row] = list()
			if tstNegs[row] is None:
				tstNegs[row] = list()
			tstLocs[row].append(col)
			tstNegs[row].append(neg)
			tstUsrs.add(row)
		tstUsrs = np.array(list(tstUsrs))
		self.tstUsrs = tstUsrs
		self.tstLocs = tstLocs
		self.tstNegs = tstNegs

	def negSampling(self):#对每个正样本生成一个负样本
		for i in range(len(self.rows)):
			#随机选择一个负样本，直到负样本与对应的用户不在稀疏矩阵中。负样本的索引存储在 self.negs 数组中
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(self.item)
				if (u, iNeg) not in self.dokmat and (u, iNeg) not in self.dokmat_trn:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.tstUsrs)

	#获取数据集中索引为 idx 的数据
	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
		#根据提供的索引 idx 返回该位置的用户和其对应的测试数据

class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data
	
	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)