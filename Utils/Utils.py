import torch as t
import torch.nn.functional as F
from Params import args
from Utils.TimeLogger import log
import numpy as np
t.autograd.set_detect_anomaly(True)

#计算了两个输入向量usrEmbeds和itmEmbeds的内积
def innerProduct(usrEmbeds, itmEmbeds):
	# 使用了PyTorch的sum函数对两个向量进行元素级相乘，并在给定维度dim=-1上对结果进行求和
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

#计算一对正样本posEmbeds和负样本negEmbeds与锚点样本ancEmbeds之间的预测差异
def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	#使用了innerProduct函数来计算锚点样本和正负样本的内积，然后返回其差异
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

#计算模型参数的正则化损失
def calcRegLoss(model):
	#遍历模型的所有参数，计算每个参数的L2范数的平方，并将这些结果相加并返回
	ret = 0
	for W in model.parameters():#model.parameters()返回模型中所有可学习参数的迭代器
		ret += W.norm(2).square()#norm(2)表示计算L2范数
	return ret

#根据输入的bprLossDiff计算奖励，主要用于排序任务的奖励计算
def calcReward(bprLossDiff, keepRate):
	#根据keepRate保留比例，选取排名靠前的一部分作为正例，构建奖励矩阵
	_, posLocs = t.topk(bprLossDiff, int(bprLossDiff.shape[0] * (1 - keepRate)))
	#topk函数在bprLossDiff张量中找到前int(bprLossDiff.shape[0] * (1 - keepRate))个最大值，并返回这些最大值的索引。
	#_是用作忽略器的变量，因为topk函数返回最大值本身和对应的索引。
	reward = t.zeros_like(bprLossDiff).cuda()#创建一个与bprLossDiff形状相同的全零张量
	reward[posLocs] = 1.0#将reward张量中posLocs索引位置处的值设置为1.0,用于表示这些位置是应该获得奖励的位置
	return reward

#计算模型参数梯度的范数
def calcGradNorm(model):
	#遍历模型的所有参数，计算每个参数梯度的L2范数，将结果平方后相加，最后返回平方根作为梯度的范数
	ret = 0
	for p in model.parameters():
		if p.grad is not None:#检查参数的梯度是否存在
			ret += p.grad.data.norm(2).square()
	ret = (ret ** 0.5)
	ret.detach()#从计算图中分离张量，使其独立于梯度计算
	return ret

#计算了对比损失，一种用于学习嵌入向量的损失计算方法
def contrastLoss(embeds1, embeds2, nodes, temp):
	#首先对输入的嵌入向量进行归一化，然后计算对比损失的分子和分母，并返回负对数似然损失值。
	embeds1 = F.normalize(embeds1, p=2)#规范化为单位L2范数（即标准化向量）
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]# u'  从 embeds1 中提取特定节点 nodes 的嵌入向量。
	pckEmbeds2 = embeds2[nodes]# u"
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)#计算了两个嵌入向量的点积，并经过指定的温度参数后进行指数操作
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)#计算了两组嵌入向量之间的点积和，并经过指定的温度参数后进行指数操作再求和。
	return -t.log(nume / deno).mean()#这个表达式包含了 nume 和 deno 的比值的负对数，并取平均值
