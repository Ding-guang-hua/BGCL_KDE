import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import scipy.sparse as sp
from Params import args
from Model import Model, Denoise, GaussianDiffusion
from DataHandler import DataHandler
import numpy as np
import pickle
from Utils.Utils import *
import os
import random
from sklearn.metrics import auc
import pynvml
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
class Coach:
	def __init__(self, handler):
		self.handler = handler

		print('USER', args.user, 'ITEM', args.item)
		print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
		self.metrics = dict()
		mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
		for met in mets:
			self.metrics['Train' + met] = list()
			self.metrics['Test' + met] = list()
	
	def makePrint(self, name, ep, reses, save):
		ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
		for metric in reses:
			val = reses[metric]
			ret += '%s = %.4f, ' % (metric, val)
			tem = name + metric
			if save and tem in self.metrics:
				self.metrics[tem].append(val)
		ret = ret[:-2] + '  '
		return ret
	
	def run(self):
		self.prepareModel()#59
		log('Model Prepared')
		log('Model Initialized')

		recallMax = 0
		ndcgMax = 0
		bestEpoch = 0
		ROC_DTI = []
		PR_DTI = []

		for ep in range(0, args.epoch):
			tstFlag = (ep % args.tstEpoch == 0)
			reses = self.trainEpoch()#75
			log(self.makePrint('Train', ep, reses, tstFlag))
			if tstFlag:
				reses = self.testEpoch()
				if args.drug_pattern == 0:
					if (reses['Recall'] > recallMax):
						recallMax = reses['Recall']
						ndcgMax = reses['NDCG']
						bestEpoch = ep
				else:
					ROC_DTI.append(reses['AUROC'])
					PR_DTI.append(reses['AUPR'])
				log(self.makePrint('Test', ep, reses, tstFlag))
			print()
		if args.drug_pattern == 0:
			print('Best epoch : ', bestEpoch, ' , Recall : ', recallMax, ' , NDCG : ', ndcgMax)
		else:
			AUROC = np.mean(ROC_DTI)
			AUPR = np.mean(PR_DTI)
			AUROC_std = np.std(ROC_DTI)
			AUPR_std = np.std(PR_DTI)
			print('AUROC : ', AUROC, '(', AUROC_std, '), AUPR : ', AUPR, '(', AUPR_std, ')')

	#准备模型的训练
	def prepareModel(self):
		self.model = Model(self.handler).cuda()
		
		self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)

		
		self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max, args.steps).cuda()
		out_dims = eval(args.dims) + [args.entity_n]
		in_dims = out_dims[::-1]
		
		self.denoise_model = Denoise(in_dims, out_dims, args.d_emb_size, norm=args.norm).cuda()
		self.denoise_opt = torch.optim.Adam(self.denoise_model.parameters(), lr=args.lr, weight_decay=0)

	
	def trainEpoch(self):
		trnLoader = self.handler.trnLoader
		trnLoader.dataset.negSampling()
		
		epLoss, epRecLoss, epClLoss = 0, 0, 0
		epDiLoss, epUKLoss = 0, 0
		steps = trnLoader.dataset.__len__() // args.batch

		diffusionLoader = self.handler.diffusionLoader

		for i, batch in enumerate(diffusionLoader):#在迭代过程中同时获取索引和元素
			batch_item, batch_index = batch#获取每个批次的 batch_item 和 batch_index
			batch_item, batch_index = batch_item.cuda(), batch_index.cuda()

			ui_matrix = self.handler.ui_matrix#使用 buildUIMatrix 方法从 trnMat 构建 PyTorch 稀疏矩阵 ui_matrix
			iEmbeds = self.model.getEntityEmbeds().detach()
			uEmbeds = self.model.getUserEmbeds().detach()

			self.denoise_opt.zero_grad()#清零优化器中存储的先前的梯度信息

			#计算训练损失
			diff_loss, ukgc_loss = self.diffusion_model.training_losses(self.denoise_model, batch_item, ui_matrix, uEmbeds, iEmbeds, batch_index)
			#KG扩散的损失   ELBO损失                       CKGC损失
			loss = diff_loss.mean() * (1-args.e_loss) + ukgc_loss.mean() * args.e_loss

			epDiLoss += diff_loss.mean().item()#计算ELBO损失的均值，并将其转换为标量值
			epUKLoss += ukgc_loss.mean().item()#CKGC损失

			loss.backward()#执行反向传播

			self.denoise_opt.step()#执行优化器 self.denoise_opt 的参数更新步骤，根据计算得到的梯度来更新模型参数

			log('Diffusion Step %d/%d' % (i, diffusionLoader.dataset.__len__() // args.batch), save=False, oneline=True)

		log('')
		log('Start to re-build kg')

		#处理输入数据，进行去噪操作，并最终生成一个表示去噪后图数据的张量
		with torch.no_grad():#上下文管理器，该上下文管理器用于禁止 PyTorch 在其中的代码段中进行梯度计算
			denoised_edges = []#存储去噪后的边缘数据
			h_list = []#存储处理后的数据 h 的列表
			t_list = []#存储尾实体t的列表

			for _, batch in enumerate(diffusionLoader):
				batch_item, batch_index = batch
				batch_item, batch_index = batch_item.cuda(), batch_index.cuda()
				iEmbeds = self.model.getEntityEmbeds().detach()
				denoised_batch = self.diffusion_model.p_sample(self.denoise_model, batch_item, args.sampling_steps, batch_index, iEmbeds)
				#denoised_batch = self.diffusion_model.p_sample(self.denoise_model, batch_item, args.sampling_steps)#实现扩散过程中的样本抽取
				#print("rebuild_k----",args.rebuild_k)
				top_item, indices_ = torch.topk(denoised_batch, k=args.rebuild_k)#找到张量 denoised_batch 中的前 args.rebuild_k 个最大值和它们对应的索引。
				for i in range(batch_index.shape[0]):
					for j in range(indices_[i].shape[0]):
						h_list.append(batch_index[i])
						t_list.append(indices_[i][j])

			edge_set = set()
			for index in range(len(h_list)):
				edge_set.add((int(h_list[index].cpu().numpy()), int(t_list[index].cpu().numpy())))
			for index in range(len(h_list)):
				if (int(t_list[index].cpu().numpy()), int(h_list[index].cpu().numpy())) not in edge_set:
					h_list.append(t_list[index])
					t_list.append(h_list[index])

			relation_dict = self.handler.relation_dict
			for index in range(len(h_list)):
				try:
					denoised_edges.append([h_list[index], t_list[index], relation_dict[int(h_list[index].cpu().numpy())][int(t_list[index].cpu().numpy())]])
				except Exception:
					continue
			graph_tensor = torch.tensor(denoised_edges)
			index_ = graph_tensor[:, :-1]
			type_ = graph_tensor[:, -1]
			denoisedKG = (index_.t().long().cuda(), type_.long().cuda())
		
		log('KG built!')

		#对生成的图数据 denoisedKG 的进一步处理和筛选
		with torch.no_grad():
			index_, type_ = denoisedKG#从 denoisedKG 中获取索引和类型数据，赋值给 index_ 和 type_
			mask = ((torch.rand(type_.shape[0]) + args.keepRate).floor()).type(torch.bool)#将随机生成的值和 args.keepRate 相加，结果向下取整
			denoisedKG = (index_[:, mask], type_[mask])
			self.generatedKG = denoisedKG

		for i, tem in enumerate(trnLoader):#遍历了数据加载器 trnLoader 中的每个 batch 数据
			ancs, poss, negs = tem
			ancs = ancs.long().cuda()#锚点
			poss = poss.long().cuda()#正样本
			negs = negs.long().cuda()#负样本

			self.opt.zero_grad()

			if args.cl_pattern == 0:
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, denoisedKG)#调用在 Model 类的前向传播函数中会处理传入的输入数据
			else:
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj)
			ancEmbeds = usrEmbeds[ancs]
			posEmbeds = itmEmbeds[poss]
			negEmbeds = itmEmbeds[negs]

			scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)#计算正例嵌入 posEmbeds 和负例嵌入 negEmbeds 之间的得分差异，通常用于推荐系统中的排名学习
			bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch#BPR损失
			regLoss = calcRegLoss(self.model) * args.reg#计算模型的正则化损失

			if args.cl_pattern == 0:
				usrEmbeds_kg, itmEmbeds_kg = self.model(self.handler.torchBiAdj)
			else:
				usrEmbeds_kg, itmEmbeds_kg = self.model(self.handler.torchBiAdj, denoisedKG)

			denoisedKGEmbeds = torch.concat([usrEmbeds, itmEmbeds], axis=0)#将用户嵌入 usrEmbeds 和物品嵌入 itmEmbeds 沿着 axis=0 的方向进行张量拼接
			kgEmbeds = torch.concat([usrEmbeds_kg, itmEmbeds_kg], axis=0)

			#对比损失CL

			clLoss = (contrastLoss(kgEmbeds[args.user:], denoisedKGEmbeds[args.user:], poss, args.temp) + contrastLoss(kgEmbeds[:args.user], denoisedKGEmbeds[:args.user], ancs, args.temp)) * args.ssl_reg

			loss = bprLoss + regLoss + clLoss#总损失 loss 是由二元分类损失 (bprLoss)、正则化损失 (regLoss) 和对比损失 (clLoss) 三部分组成的加和

			epLoss += loss.item()#将当前批次的总损失 loss 转换为 Python 数值，并累加到一个叫做 epLoss 的损失指标中，用于跟踪整个训练过程中的总损失值。
			epRecLoss += bprLoss.item()#将当前批次的二元分类损失 bprLoss 转换为 Python 数值，并累加到 epRecLoss 中，用于跟踪整个训练过程中的二元分类损失值。
			epClLoss += clLoss.item()#将当前批次的对比损失 clLoss 转换为 Python 数值，并累加到 epClLoss 中，用于跟踪整个训练过程中的对比损失值。

			loss.backward()#反向传播
			self.opt.step()#优化器参数更新

			log('Step %d/%d: loss = %.3f, regLoss = %.3f' % (i, steps, loss, regLoss), save=False, oneline=True)

		ret = dict()
		ret['Loss'] = epLoss / steps#总损失的平均值
		ret['recLoss'] = epRecLoss / steps#BPR
		ret['clLoss'] = epClLoss / steps#对比损失CL的平均值
		ret['diLoss'] = epDiLoss / diffusionLoader.dataset.__len__()#ELBO
		ret['UKGCLoss'] = epUKLoss / diffusionLoader.dataset.__len__()#CKGC
		return ret

	def testEpoch(self):
		tstLoader = self.handler.tstLoader# 获取测试数据加载器
		epRecall, epNdcg = [0] * 2# 初始化召回率和NDCG为零
		epAUROC, epAUPR = [0] * 2  # 初始化AUROC和AUPR为零
		i = 0
		num = tstLoader.dataset.__len__()# 获取测试数据集的大小
		steps = num // args.tstBat# 计算步数

		with torch.no_grad():
			if args.cl_pattern == 0:
				denoisedKG = self.generatedKG
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, mess_dropout=False, kg=denoisedKG)
			else:
				usrEmbeds, itmEmbeds = self.model(self.handler.torchBiAdj, mess_dropout=False)

		for usr, trnMask in tstLoader:## 对于测试数据加载器中的每个用户和训练掩码
			i += 1
			usr = usr.long().cuda()
			trnMask = trnMask.cuda()

			#矩阵乘法操作，其中 usrEmbeds[usr] 是对应于用户索引 usr 的嵌入向量，而 itmEmbeds 是物品的嵌入向量。通过将用户嵌入向量乘以物品嵌入向量的转置，得到用户与所有物品之间的预测结果。
			#* (1 - trnMask): 这个步骤将训练掩码应用到预测结果上，以便在预测时忽略已经在训练集中出现过的项目。
			#- trnMask * 1e8: 在这一步中，为了避免模型预测已经在训练集中出现的项目，对训练集中出现过的项目的预测打分进行调整，将它们的分数减去一个较大的数值，这里是1e8。
			allPreds = t.mm(usrEmbeds[usr], t.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
			_, topLocs = t.topk(allPreds, args.topk)## 获取Top K 推荐列表
			if args.drug_pattern == 0:
				recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)# 239 计算召回率和NDCG
				epRecall += recall# 累积召回率
				epNdcg += ndcg# 累积 NDCG
				log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
			else:
				roc_auc, aupr = self.calcAUC(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, self.handler.tstLoader.dataset.tstNegs, usr)
				epAUROC += roc_auc  # 累积AUROC
				epAUPR += aupr  # 累积AUPR
				log('Steps %d/%d: auroc = %.2f, aupr = %.2f          ' % (i, steps, roc_auc, aupr), save=False,
					oneline=True)
		ret = dict()
		if args.drug_pattern == 0:
			ret['Recall'] = epRecall / num# 计算平均召回率
			ret['NDCG'] = epNdcg / num# 计算平均 NDCG
		else:
			ret['AUROC'] = epAUROC  # 计算AUROC
			ret['AUPR'] = epAUPR  # 计算平均AUPR
		return ret

	#计算模型在给定用户批次上的召回率和NDCG指标
	def calcRes(self, topLocs, tstLocs, batIds):
		assert topLocs.shape[0] == len(batIds)# 确保 topLocs 和 batIds 的形状匹配
		allRecall = allNdcg = 0# 初始化总召回率和总NDCG为零
		for i in range(len(batIds)):# 遍历每个批次的用户
			temTopLocs = list(topLocs[i])# 获取该用户的推荐物品列表
			temTstLocs = tstLocs[batIds[i]]# 获取该用户在测试集上真实的物品列表
			tstNum = len(temTstLocs)# 获取真实物品列表的长度
			maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])# 计算最大DCG值
			recall = dcg = 0
			for val in temTstLocs:# 遍历真实物品列表
				if val in temTopLocs:# 如果推荐列表中包含该物品
					recall += 1
					dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
			recall = recall / tstNum# 计算召回率
			ndcg = dcg / maxDcg# 计算NDCG值
			allRecall += recall
			allNdcg += ndcg
		return allRecall, allNdcg
	def calcAUC(self, topLocs, tstLocs, negs, batIds):
		assert topLocs.shape[0] == len(batIds)# 确保 topLocs 和 batIds 的形状匹配
		TPR =[]
		FPR = []
		PRec = []
		for i in range(len(batIds)):# 遍历每个批次的用户
			temTopLocs = list(topLocs[i])# 获取该用户的推荐物品列表
			temTstLocs = tstLocs[batIds[i]]# 获取该用户在测试集上真实的物品列表
			temNegs = negs[batIds[i]]  #实际阴性
			if not temTstLocs or not temNegs:
				continue

			tstNum = len(temTstLocs)# 获取真实物品列表的长度  TP+FN
			negNum = len(temNegs) #FP+TN

			tp = dcg = 0
			precision = fp = 0
			for val in temTstLocs:# 遍历真实物品列表  实际阳性
				if val in temTopLocs:# 如果推荐列表中包含该物品  预测阳性
					tp += 1

			if tp == 0:
				continue
			for val in temNegs:#实际阴性
				if val in temTopLocs:#  预测阳性
					fp += 1
			if tp + fp == 0:
				precision = 0
			else:
				precision = tp / (tp + fp)
			tpr = tp / tstNum# 计算召回率
			fpr = fp / negNum

			TPR.append(tpr)
			FPR.append(fpr)
			PRec.append(precision)

		TPR.append(1.0)
		FPR.append(1.0)
		PRec.append(1.0)
		TPR.append(0)
		FPR.append(0)
		PRec.append(0)
		FPR_sorted = sorted(FPR)
		TPR_sorted = sorted(TPR)
		PR_sorted = sorted(PRec)
		# 计算 AUROC
		roc_auc = auc(FPR_sorted, TPR_sorted)
		aupr = auc(TPR_sorted, PR_sorted)

		return roc_auc, aupr
def seed_it(seed):
	random.seed(seed)#设置 Python 的随机数生成器的种子
	os.environ["PYTHONSEED"] = str(seed)
	np.random.seed(seed)# NumPy 的随机数生成器的种子
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True 
	torch.backends.cudnn.enabled = True
	torch.manual_seed(seed)

if __name__ == '__main__':
	torch.cuda.set_device(1)
	seed_it(args.seed)

	#os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
	logger.saveDefault = True
	
	log('Start')
	handler = DataHandler()
	handler.LoadData()
	log('Load Data')

	decice_idx = torch.cuda.current_device()
	print(f"GPU---{decice_idx}")
	coach = Coach(handler)
	coach.run()

