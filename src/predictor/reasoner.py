import torch
from torch import nn
from torch_scatter import scatter_max


class MatrixReasoner(nn.Module):

	def __init__(self, matrix_path, n_entities, n_relations, device):
		super().__init__()

		self.rels = [None] * n_relations
		for i in range(n_relations):
			self.rels[i] = torch.load(matrix_path+'/P_'+str(i)+'.pt', map_location=torch.device('cpu')).to(device)

		self.n_entities = n_entities
		self.n_relations = n_relations

		self.device = device

	def forward(self, emb_vec, rel_id):
		supp = (emb_vec.to(self.device) * self.rels[rel_id].T).T.coalesce()
		res = scatter_max(supp.values(), supp.indices()[1], dim_size=self.n_entities)[0]
		return res

	def update(self, v1, v2, aggr, conorm=False):
		if aggr == 'max':
			if conorm:
				return torch.maximum(v1,v2)
			else:
				return torch.minimum(v1,v2)
		elif aggr == 'prod':
			if conorm:
				return 1 - (1 - v1) * (1 - v2)
			else:
				return v1 * v2
		else:
			raise NotImplementedError

	def process(self, graph, node_id, aggr = 'max'):

		node_info = graph.nodes[node_id]
		# Deal with constant nodes
		if node_info['type'] == 'const':
			emb_vec = torch.zeros(self.n_entities, device = self.device)
			emb_vec[node_info['idx']] = 1
			return emb_vec
	
		# Otherwise, initialise the embedding as all-1's
		emb_vec = torch.ones(self.n_entities, device = self.device)
		dis_part = torch.zeros(self.n_entities, device = self.device)
		used_dis = False

		# Process information from all neighbours
		for v in graph[node_id]:
			rel_info = graph[node_id][v]
			emb_v = self.process(graph, v, aggr)
			propagated = self(emb_v, rel_info['idx'])
			if rel_info['neg']:
				propagated = 1 - propagated
			
			if 'dis_part' in rel_info:
				dis_part = self.update(dis_part, propagated, aggr, conorm=True)
				used_dis = True
			else:
				emb_vec = self.update(emb_vec, propagated, aggr)

		if used_dis:
			emb_vec = self.update(emb_vec, dis_part, aggr)

		return emb_vec