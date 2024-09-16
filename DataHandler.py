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

	
	def loadOneFile(self, filename):
		with open(filename, 'rb') as fs:
			ret = (pickle.load(fs) != 0).astype(np.float32)
		if type(ret) != coo_matrix:
			ret = sp.coo_matrix(ret)

		return ret

	
	def readTriplets(self, file_name):
		can_triplets_np = np.loadtxt(file_name, dtype=np.int32)
		can_triplets_np = np.unique(can_triplets_np, axis=0)

		inv_triplets_np = can_triplets_np.copy()
		inv_triplets_np[:, 0] = can_triplets_np[:, 2]
		inv_triplets_np[:, 2] = can_triplets_np[:, 0]
		inv_triplets_np[:, 1] = can_triplets_np[:, 1] + max(can_triplets_np[:, 1]) + 1
		triplets = np.concatenate((can_triplets_np, inv_triplets_np), axis=0)

		n_relations = max(triplets[:, 1]) + 1

		args.relation_num = n_relations

		args.entity_n = max(max(triplets[:, 0]), max(triplets[:, 1])) + 1

		return triplets
	
	
	def buildGraphs(self, triplets):
		kg_dict = defaultdict(list)
        # h, t, r
		kg_edges = list()

		print("Begin to load knowledge graph triples ...")

		kg_counter_dict = {}
		
		for h_id, r_id, t_id in tqdm(triplets, ascii=True):
			if h_id not in kg_counter_dict.keys():
				kg_counter_dict[h_id] = set()
			if t_id not in kg_counter_dict[h_id]:
				kg_counter_dict[h_id].add(t_id)
			else:
				continue
			kg_edges.append([h_id, t_id, r_id])
			kg_dict[h_id].append((r_id, t_id))

		return kg_edges, kg_dict
	
	
	def buildKGMatrix(self, kg_edges):
		edge_list = []
		for h_id, t_id, r_id in kg_edges:
			edge_list.append((h_id, t_id))
		edge_list = np.array(edge_list)
		
		kgMatrix = sp.csr_matrix((np.ones_like(edge_list[:,0]), (edge_list[:,0], edge_list[:,1])), dtype='float64', shape=(args.entity_n, args.entity_n))

		return kgMatrix

	
	def normalizeAdj(self, mat, S_values):
		degree = np.array(mat.sum(axis=-1)) 
		dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
		dInvSqrt[np.isinf(dInvSqrt)] = 0.0
		dInvSqrtMat = sp.diags(dInvSqrt)
		mat = mat + S_values
		#print("Effectiveness of Attention-aware Matrix")
		return dInvSqrtMat.transpose().dot(mat).dot(dInvSqrtMat).tocoo()
		


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
	
	def calculate_S_from_adjacency_matrix(self, adj_matrix, L, M, k, epsilon):
		print("开始构建相似度矩阵")
		S_values = np.zeros(adj_matrix.shape)
		kadj_set = self.k_adjacency_matrix(adj_matrix, k)

		for u in range(adj_matrix.shape[0]):
			adj_matrix_u = adj_matrix.getrow(u).toarray()[0]
			for v in range(adj_matrix.shape[1]):
				if v > u:
					break
				if adj_matrix[u, v] == 0:  
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
				
				u_flat = set([node for seq in u_m_l for node in seq])
				v_flat = set([node for seq in v_m_l for node in seq])
				print("-----------------",len(u_flat),len(v_flat))

				
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
				
		S_values = csr_matrix(S_values)
		with open(self.predir+'Similarity.pkl', 'wb') as file:
			pickle.dump(S_values, file)
		print(type(S_values))
		return S_values

	
	def makeTorchAdj(self, mat):
		# make ui adj
		a = sp.csr_matrix((args.user, args.user))
		b = sp.csr_matrix((args.item, args.item))
		
		mat = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
		mat = (mat != 0) * 1.0
		
		#self.calculate_S_from_adjacency_matrix(mat, args.L, args.M, args.k, args.epsilon)

		with open(self.S_file, 'rb') as file:
			 S_values= pickle.load(file)
		print("S_values-----",self.S_file,args.epsilon)

		S_values = args.epsilon * S_values
		mat = (mat + sp.eye(mat.shape[0])) * 1.0

		mat = self.normalizeAdj(mat, S_values)


		# make cuda tensor
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	
	def RelationDictBuild(self):
		relation_dict = {}
		
		for head in self.kg_dict:
			relation_dict[head] = {}
			for (relation, tail) in self.kg_dict[head]:
				relation_dict[head][tail] = relation
		return relation_dict

	
	def buildUIMatrix(self, mat):
		idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
		vals = torch.from_numpy(mat.data.astype(np.float32))
		shape = torch.Size(mat.shape)
		return torch.sparse.FloatTensor(idxs, vals, shape).cuda()

	
	def LoadData(self):
		trnMat = self.loadOneFile(self.trnfile)
		tstMat = self.loadOneFile(self.tstfile)
		self.trnMat = trnMat

		args.user, args.item = trnMat.shape
		#print("-----------------",args.user,args.item)
		
		self.torchBiAdj = self.makeTorchAdj(trnMat)

		self.ui_matrix = self.buildUIMatrix(trnMat)

		trnData = TrnData(trnMat)
		
		self.trnLoader = dataloader.DataLoader(trnData, batch_size=args.batch, shuffle=True, num_workers=0)
		tstData = TstData(tstMat, trnMat)
		self.tstLoader = dataloader.DataLoader(tstData, batch_size=args.tstBat, shuffle=False, num_workers=0)

		kg_triplets = self.readTriplets(self.kgfile)
		
		self.kg_edges, self.kg_dict = self.buildGraphs(kg_triplets)

		self.kg_matrix = self.buildKGMatrix(self.kg_edges)
		print("kg shape: ", self.kg_matrix.shape)
		print("number of edges in KG: ", len(self.kg_edges))
		
		self.diffusionData = DiffusionData(torch.FloatTensor(self.kg_matrix.A))
		self.diffusionLoader = dataloader.DataLoader(self.diffusionData, batch_size=args.batch, shuffle=True, num_workers=0)

		self.relation_dict = self.RelationDictBuild()


class TrnData(data.Dataset):
	
	def __init__(self, coomat):
		self.rows = coomat.row
		self.cols = coomat.col
		self.dokmat = coomat.todok()
		self.negs = np.zeros(len(self.rows)).astype(np.int32)

	def negSampling(self):
		for i in range(len(self.rows)):
			
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


class TstData(data.Dataset):
	def __init__(self, coomat, trnMat):
		self.csrmat = (trnMat.tocsr() != 0) * 1.0
		self.dokmat = coomat.todok()  
		self.dokmat_trn = trnMat.todok()
		self.rows = coomat.row
		self.negs = np.zeros(len(self.rows)).astype(np.int32)
		self.item = np.max(coomat.col)
		self.negSampling()

		tstLocs = [None] * coomat.shape[0]
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

	def negSampling(self):
		for i in range(len(self.rows)):
			
			u = self.rows[i]
			while True:
				iNeg = np.random.randint(self.item)
				if (u, iNeg) not in self.dokmat and (u, iNeg) not in self.dokmat_trn:
					break
			self.negs[i] = iNeg

	def __len__(self):
		return len(self.tstUsrs)

	
	def __getitem__(self, idx):
		return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
		

class DiffusionData(data.Dataset):
	def __init__(self, data):
		self.data = data
	
	def __getitem__(self, index):
		item = self.data[index]
		return item, index
	
	def __len__(self):
		return len(self.data)
