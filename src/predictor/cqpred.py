import torch
from torch import nn

from .complex import ComplEx
from .perfect import PerfeCT

def subpredictor_from_config(pred_config, device):
	predictor = None

	if pred_config['type'] == 'ComplEx':
		predictor = ComplEx(
			n_relations = pred_config['n_relations'],
			n_entities = pred_config['n_entities'],
			embedding_dim = pred_config['embedding_dim'],
			device = device,
			tau = pred_config['tau']
		)
		if 'load_path' in pred_config:
			predictor.load_state_dict(torch.load(pred_config['load_path'], map_location=torch.device(device)))

	elif pred_config['type'] == 'PerfeCT':
		predictor = PerfeCT(
			data_path = pred_config['graph_path'],
			device = device
		)
	return predictor


class CQPred(nn.Module):

	def __init__(self,
				config_perfect,
				config_predictor,
				device,
				scaling_rule = 'qto',
				eps = 0.0001,
				temp = 1):

		super().__init__()

		self.predictor = subpredictor_from_config(config_predictor, device)
		self.perfect = subpredictor_from_config(config_perfect, device)

		self.scaling = scaling_rule
		assert (self.scaling == 'qto' or self.scaling == 'fit')

		self.eps = eps
		self.temp = temp

		self.device = device

	def scale(self, predictor_scores, perfect_scores):
		ez_ans_size = torch.sum(perfect_scores, dim=1).view(-1, 1)
		ez_ans_size[ez_ans_size < 1] = 1
		if self.scaling == 'qto':
			return predictor_scores * ez_ans_size
		normalisation_term = [predictor_scores[i][perfect_scores[i] > 0].sum() for i in range(predictor_scores.shape[0])]
		normalisation_term = torch.Tensor(normalisation_term).view(-1,1).to(predictor_scores.device)
		normalisation_term[normalisation_term==0] = 1
		return predictor_scores * ez_ans_size / normalisation_term
	
	def forward(self, heads, rels, tails):
		
		tail_scores = self.tail_scores(heads, rels)
		return tail_scores[torch.arange(tail_scores.shape[0]), tails.long()]
	
	def tail_scores(self, heads, rels):
		
		with torch.no_grad():
			# Retrieve the knowledge from the KG
			perf_response = self.perfect.tail_scores(heads, rels)
			perf_response[perf_response > 0] = 1
			perf_response[perf_response < 0.5] = 0

			# Generate and scale predictor scores
			pred_response = self.predictor.tail_scores(heads, rels) * self.temp
			pred_response = pred_response.softmax(dim=1)
			scaled_scores = self.scale(pred_response, perf_response)
			scaled_scores = torch.clip(scaled_scores, 0, 1-self.eps)

		return torch.clip(torch.maximum(perf_response, scaled_scores), 0, 1)
		
	def head_scores(self, tails, rels):
		rels = rels - 2 * (rels % 2) + 1
		return self.tail_scores(tails, rels)






class BinCQPred(nn.Module):

	def __init__(self,
				config_perfect,
				config_predictor,
				device,
				scaling_rule = 'qto',
				eps = 0.0001,
				temp = 1,
				thr = 0.5):

		super().__init__()

		self.predictor = subpredictor_from_config(config_predictor, device)
		self.perfect = subpredictor_from_config(config_perfect, device)

		self.scaling = scaling_rule
		assert (self.scaling == 'qto' or self.scaling == 'fit')

		self.eps = eps
		self.temp = temp
		self.thr = thr

		self.device = device

	def scale(self, predictor_scores, perfect_scores):
		ez_ans_size = torch.sum(perfect_scores, dim=1).view(-1, 1)
		ez_ans_size[ez_ans_size < 1] = 1
		if self.scaling == 'qto':
			return predictor_scores * ez_ans_size
		normalisation_term = [predictor_scores[i][perfect_scores[i] > 0].sum() for i in range(predictor_scores.shape[0])]
		normalisation_term = torch.Tensor(normalisation_term).view(-1,1).to(predictor_scores.device)
		normalisation_term[normalisation_term==0] = 1
		return predictor_scores * ez_ans_size / normalisation_term
	
	def forward(self, heads, rels, tails):
		
		tail_scores = self.tail_scores(heads, rels)
		return tail_scores[torch.arange(tail_scores.shape[0]), tails.long()]
	
	def tail_scores(self, heads, rels):
		
		with torch.no_grad():
			# Retrieve the knowledge from the KG
			perf_response = self.perfect.tail_scores(heads, rels)
			perf_response[perf_response > 0] = 1
			perf_response[perf_response < 0.5] = 0

			# Generate and scale predictor scores
			pred_response = self.predictor.tail_scores(heads, rels) * self.temp
			pred_response = pred_response.softmax(dim=1)
			scaled_scores = self.scale(pred_response, perf_response)
			scaled_scores = torch.clip(scaled_scores, 0, 1-self.eps)

		res = torch.clip(torch.maximum(perf_response, scaled_scores), 0, 1)
		res[res < self.thr] = -1
		res[res > 0] = 1
		return res
		
	def head_scores(self, tails, rels):
		rels = rels - 2 * (rels % 2) + 1
		return self.tail_scores(tails, rels)



class SignCQPred(nn.Module):

	def __init__(self,
				config_perfect,
				config_predictor,
				device,
				scaling_rule = 'qto',
				eps = 0.0001,
				temp = 1,
				thr = 0.5):

		super().__init__()

		self.predictor = subpredictor_from_config(config_predictor, device)
		self.perfect = subpredictor_from_config(config_perfect, device)

		self.scaling = scaling_rule
		assert (self.scaling == 'qto' or self.scaling == 'fit')

		self.eps = eps
		self.temp = temp
		self.thr = thr

		self.device = device

	def scale(self, predictor_scores, perfect_scores):
		ez_ans_size = torch.sum(perfect_scores, dim=1).view(-1, 1)
		ez_ans_size[ez_ans_size < 1] = 1
		if self.scaling == 'qto':
			return predictor_scores * ez_ans_size
		normalisation_term = [predictor_scores[i][perfect_scores[i] > 0].sum() for i in range(predictor_scores.shape[0])]
		normalisation_term = torch.Tensor(normalisation_term).view(-1,1).to(predictor_scores.device)
		normalisation_term[normalisation_term==0] = 1
		return predictor_scores * ez_ans_size / normalisation_term
	
	def forward(self, heads, rels, tails):
		tail_scores = self.tail_scores(heads, rels)
		return tail_scores[torch.arange(tail_scores.shape[0]), tails.long()]
	
	def tail_scores(self, heads, rels):
		
		with torch.no_grad():
			# Retrieve the knowledge from the KG
			perf_response = self.perfect.tail_scores(heads, rels)
			perf_response[perf_response > 0] = 1
			perf_response[perf_response < 0.5] = 0

			# Generate and scale predictor scores
			pred_response = self.predictor.tail_scores(heads, rels) * self.temp
			pred_response = pred_response.softmax(dim=1)
			scaled_scores = self.scale(pred_response, perf_response)
			scaled_scores = torch.clip(scaled_scores, 0, 1-self.eps)

		res = torch.clip(torch.maximum(perf_response, scaled_scores), 0, 1)
		res -= self.thr
		res *= 2
		return res
		
	def head_scores(self, tails, rels):
		rels = rels - 2 * (rels % 2) + 1
		return self.tail_scores(tails, rels)
	
	def to(self, device):
		self.predictor = self.predictor.to(device)
		self.predictor.device = device
		self.perfect.to(device)
		self.device = device

		return self


class SymCQPred(nn.Module):

	def __init__(self,
			config_perfect,
			config_predictor,
			logDelta_path,
			device,
			eps = 1e-4,
			temp = 1,
			thr = 0.5
		):
		
		super().__init__()

		self.predictor = subpredictor_from_config(config_predictor, device)
		self.perfect = subpredictor_from_config(config_perfect, device)

		self.temp = temp
		self.logDelta = torch.load(logDelta_path)
		self.logDelta = self.logDelta.to(device)

		self.eps = eps
		self.thr = thr

		self.device = device

	def forward(self, heads, rels, tails):		
		tail_scores = self.tail_scores(heads, rels)
		return tail_scores[torch.arange(tail_scores.shape[0]), tails.long()]

	def head_scores(self, tails, rels):
		rels = rels - 2 * (rels % 2) + 1
		return self.tail_scores(tails, rels)
	
	def tail_scores(self, heads, rels):
		
		inv_rels = rels + 1 - 2 * (rels % 2)

		with torch.no_grad():
			# Retrieve the knowledge from the KG
			perf_response = self.perfect.tail_scores(heads, rels)
			perf_response[perf_response > 0] = 1
			perf_response[perf_response < 0.5] = 0

			# Generate and scale predictor scores
			direct_scores  = self.temp * self.predictor.tail_scores(heads,rels) + self.logDelta[rels.long(), heads.long()].view(-1,1)
			reverse_scores = self.temp * self.predictor.head_scores(heads,inv_rels) + self.logDelta[inv_rels]
			scaled_scores = torch.clip(torch.maximum(direct_scores.exp(), reverse_scores.exp()), 0, 1-self.eps)

		scores = torch.maximum(perf_response, scaled_scores)
		scores = (scores - 0.5) * 2

		return scores