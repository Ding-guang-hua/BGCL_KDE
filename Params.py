import argparse

def ParseArgs():#用于解析命令行参数并返回相应的参数对象
	parser = argparse.ArgumentParser(description='Model Params')
	parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')#学习率
	parser.add_argument('--batch', default=1024, type=int, help='batch size')#批量大小
	parser.add_argument('--kg_batch', default=4096, type=int, help='batch size for kg')#用于知识图谱的批量大小
	parser.add_argument('--tstBat', default=1024, type=int, help='number of users in a testing batch')#测试批量中的用户数量
	parser.add_argument('--reg', default=1e-7, type=float, help='weight decay regularizer')#权重衰减正则化器
	parser.add_argument('--epoch', default=25, type=int, help='number of epochs')#训练的时期数量 机器学习中的迭代训练过程中的一个完整的数据集被用于训练的次数
	parser.add_argument('--save_path', default='tem', help='file name to save model and training record')#保存模型和训练记录的文件名
	parser.add_argument('--latdim', default=64, type=int, help='embedding size')#嵌入大小
	parser.add_argument('--gnn_layer', default=2, type=int, help='number of gnn layers')#GNN 层的数量
	parser.add_argument('--load_model', default=None, help='model name to load')#要加载的模型名称
	parser.add_argument('--topk', default=20, type=int, help='K of top K')#前 K 的值
	parser.add_argument('--data', default='yelp2018', type=str, help='name of dataset')# 数据集名称
	parser.add_argument('--tstEpoch', default=1, type=int, help='number of epoch to test while training')#训练时用于测试的时期数量
	parser.add_argument('--gpu', default='1', type=str, help='indicates which gpu to use')#指示要使用的 GPU 编号
	parser.add_argument('--layer_num_kg', default=1, type=int)#kg 层的数量
	parser.add_argument('--mess_dropout_rate', default=0.1, type=float)#消息 dropout 率
	parser.add_argument('--ssl_reg', default=1e-1, type=float, help='weight for contrative learning')#对比学习的权重
	parser.add_argument('--temp', default=0.1, type=float, help='temperature in contrastive learning')#对比学习中的温度
	parser.add_argument("--seed", type=int, default=421, help="random seed")# 随机种子

	parser.add_argument('--dims', type=str, default='[1000]')#维度数组
	parser.add_argument('--d_emb_size', type=int, default=10)#嵌入大小
	parser.add_argument('--norm', type=bool, default=True)# 是否规范化
	parser.add_argument('--steps', type=int, default=5)#步骤数量
	parser.add_argument('--noise_scale', type=float, default=0.1)#噪音比例
	parser.add_argument('--noise_min', type=float, default=0.0001)#最小噪声值
	parser.add_argument('--noise_max', type=float, default=0.02)#最大噪声值
	parser.add_argument('--sampling_steps', type=int, default=0)#采样步骤数

	parser.add_argument('--rebuild_k', type=int, default=1)# 重建 K 值
	parser.add_argument('--e_loss', type=float, default=0.5)#E 损失值

	parser.add_argument('--keepRate', type=float, default=0.5)#保留率
	parser.add_argument('--res_lambda', type=float, default=0.5)#lambda 值
	parser.add_argument('--triplet_num', type=int, default=10)#三元组数量
	parser.add_argument('--cl_pattern', type=int, default=0)#对比学习模式

	parser.add_argument('--similarity', type=str, default='80_10_3', help='Attention-aware Matrix')
	parser.add_argument('--L', type=int, default=80, help='Number of steps in each random walk')
	parser.add_argument('--M', type=int, default=10, help='Number of random walks')
	parser.add_argument('--k', type=int, default=3, help='Value of k for k-nearest neighbors')
	parser.add_argument('--epsilon', type=float, default=0.5, help='Value of epsilon for controlling S contribution')
	parser.add_argument('--drug_pattern', type=int, default=0)  # 药物发现模式

	return parser.parse_args()#解析命令行参数
args = ParseArgs()