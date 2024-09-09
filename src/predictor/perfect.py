import torch
from torch import nn
import pandas as pd


class PerfeCT(nn.Module):
	
	def __init__(self, data_path, device):
		super().__init__()

		# Read and save the edge data
		self.data = torch.from_numpy(pd.read_csv(data_path, header=None, sep='\t').to_numpy()).T.long().to(device)
		self.n_entities 	= self.data[0].max()+1
		self.n_relations 	= self.data[1].max()+1

		self.device = device
		self.to(device)

	def forward(self, heads, rels, tails):
		queries = torch.stack([heads, rels, tails]).long().T.to(self.device)
		response = torch.zeros((queries.shape[0])).to(self.device)
		for i in range(queries.shape[0]):
			response[i] = 1 if ((torch.min(self.data.T == queries[i], dim=1)[0]).max()) else 0
		response = 10*(response - 0.5)
		return response
	
	def tail_scores(self, head, rel):

		queries = torch.stack([head, rel]).long().T.to(self.device)
		response = torch.zeros((queries.shape[0], self.n_entities)).to(self.device)
		for i in range(queries.shape[0]):
			response[i][self.data[2,torch.min(self.data[:2].T==queries[i], dim=1)[0]]] = 1
		response = 10*(response - 0.5)

		return response

	def head_scores(self, tail, rel):

		queries = torch.stack([rel, tail]).long().T.to(self.device)
		response = torch.zeros((queries.shape[0], self.n_entities)).to(self.device)
		for i in range(queries.shape[0]):
			response[i][self.data[0,torch.min(self.data[1:].T==queries[i], dim=1)[0]]] = 1
		response = 10*(response - 0.5)

		return response

	def rel_scores(self, head, tail):

		queries = torch.stack([head, tail]).long().T.to(self.device)
		response = torch.zeros((queries.shape[0], self.n_relations)).to(self.device)
		for i in range(queries.shape[0]):
			response[i][self.data[1,torch.min(self.data[[0,2]].T==queries[i], dim=1)[0]]] = 1
		response = 10*(response - 0.5)

		return response
	
	def to(self, device):
		self.device = device
		self.data = self.data.to(device)
		return self