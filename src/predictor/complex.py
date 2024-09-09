import torch
from torch import nn


class ComplEx(nn.Module):

	def __init__(self, n_relations, n_entities, embedding_dim, device, tau=0):
		super().__init__()

		self.embedding_dim = embedding_dim
		self.n_entities = n_entities
		self.n_relations = n_relations

		self.rel_re = nn.Embedding(n_relations, embedding_dim)
		self.rel_im = nn.Embedding(n_relations, embedding_dim)

		self.ent_re = nn.Embedding(n_entities, embedding_dim)
		self.ent_im = nn.Embedding(n_entities, embedding_dim)

		self.sigmoid = torch.nn.functional.sigmoid

		self.device = device

		self.tau = tau

		self.rel_re.weight.data *= 1e-3
		self.rel_im.weight.data *= 1e-3
		self.ent_re.weight.data *= 1e-3
		self.ent_im.weight.data *= 1e-3

		self.to(device)

	def forward(self, heads, rels, tails):

		# Evaluate relation embeddings
		rel_re = self.rel_re(rels.to(self.device))
		rel_im = self.rel_im(rels.to(self.device))

		# Evaluate head and tail embeddings
		head_re = self.ent_re(heads.to(self.device))
		head_im = self.ent_im(heads.to(self.device))
		tail_re = self.ent_re(tails.to(self.device))
		tail_im = self.ent_im(tails.to(self.device))

		score = torch.zeros(heads.shape[0], self.embedding_dim).to(self.device)

		score += rel_re * head_re * tail_re
		score += rel_re * head_im * tail_im
		score += rel_im * head_re * tail_im
		score -= rel_im * head_im * tail_re

		score = torch.sum(score, dim=1)

		return score - self.tau

	
	def tail_scores(self, head, rel, train=False):
		
		if not train:
			self.eval()
			with torch.no_grad():
				# Evaluate embeddings of the passed head and relation
				heads_re = self.ent_re(torch.Tensor(head).long().to(self.device))
				heads_im = self.ent_im(torch.Tensor(head).long().to(self.device))

				rels_re = self.rel_re(torch.Tensor(rel).long().to(self.device))
				rels_im = self.rel_im(torch.Tensor(rel).long().to(self.device))

				# Extract embeddings for all the tensors
				tails_re = self.ent_re(torch.arange(self.n_entities).long().to(self.device)).T
				tails_im = self.ent_im(torch.arange(self.n_entities).long().to(self.device)).T

				scores = torch.mm(heads_re*rels_re, tails_re)
				scores += torch.mm(heads_im*rels_re, tails_im)
				scores += torch.mm(heads_im*rels_re, tails_im)
				scores -= torch.mm(heads_im*rels_im, tails_re)
			self.train()
		else:
			heads_re = self.ent_re(torch.Tensor(head).long().to(self.device))
			heads_im = self.ent_im(torch.Tensor(head).long().to(self.device))

			rels_re = self.rel_re(torch.Tensor(rel).long().to(self.device))
			rels_im = self.rel_im(torch.Tensor(rel).long().to(self.device))

			# Extract embeddings for all the tensors
			tails_re = self.ent_re(torch.arange(self.n_entities).long().to(self.device)).T
			tails_im = self.ent_im(torch.arange(self.n_entities).long().to(self.device)).T

			scores = torch.mm(heads_re*rels_re, tails_re)
			scores += torch.mm(heads_im*rels_re, tails_im)
			scores += torch.mm(heads_im*rels_re, tails_im)
			scores -= torch.mm(heads_im*rels_im, tails_re)

		return scores - self.tau

	def head_scores(self, tail, rel, train=False):
		
		if not train:
			self.eval()
			with torch.no_grad():
				# Evaluate embeddings of the passed head and relation
				tails_re = self.ent_re(torch.Tensor(tail).long().to(self.device))
				tails_im = self.ent_im(torch.Tensor(tail).long().to(self.device))

				rels_re = self.rel_re(torch.Tensor(rel).long().to(self.device))
				rels_im = self.rel_im(torch.Tensor(rel).long().to(self.device))

				# Extract embeddings for all the tensors
				heads_re = self.ent_re(torch.arange(self.n_entities).long().to(self.device)).T
				heads_im = self.ent_im(torch.arange(self.n_entities).long().to(self.device)).T

				scores = torch.mm(tails_re*rels_re, heads_re)
				scores += torch.mm(tails_im*rels_re, heads_im)
				scores += torch.mm(tails_im*rels_re, heads_im)
				scores -= torch.mm(tails_re*rels_im, heads_im)

			self.train()
		else:
			tails_re = self.ent_re(torch.Tensor(tail).long().to(self.device))
			tails_im = self.ent_im(torch.Tensor(tail).long().to(self.device))

			rels_re = self.rel_re(torch.Tensor(rel).long().to(self.device))
			rels_im = self.rel_im(torch.Tensor(rel).long().to(self.device))

			# Extract embeddings for all the tensors
			heads_re = self.ent_re(torch.arange(self.n_entities).long().to(self.device)).T
			heads_im = self.ent_im(torch.arange(self.n_entities).long().to(self.device)).T

			scores = torch.mm(tails_re*rels_re, heads_re)
			scores += torch.mm(tails_im*rels_re, heads_im)
			scores += torch.mm(tails_im*rels_re, heads_im)
			scores -= torch.mm(tails_re*rels_im, heads_im)


		return scores - self.tau


	def rel_scores(self, head, tail, train=False):

		if not train:
			self.eval()
			with torch.no_grad():
				# Evaluate embeddings of the passed head and tail
				tails_re = self.ent_re(torch.Tensor(tail).long().to(self.device))
				tails_im = self.ent_im(torch.Tensor(tail).long().to(self.device))
				
				heads_re = self.ent_re(torch.Tensor(head).long().to(self.device))
				heads_im = self.ent_im(torch.Tensor(head).long().to(self.device))


				# Extract embeddings for all the tensors
				rels_re = self.rel_re(torch.arange(self.n_relations).long().to(self.device)).T
				rels_im = self.rel_im(torch.arange(self.n_relations).long().to(self.device)).T

				scores = torch.mm(tails_re*heads_re, rels_re)
				scores += torch.mm(tails_im*heads_im, rels_re)
				scores += torch.mm(tails_im*heads_im, rels_re)
				scores -= torch.mm(tails_re*heads_im, rels_im)

			self.train()
		else:
			# Evaluate embeddings of the passed head and tail
			tails_re = self.ent_re(torch.Tensor(tail).long().to(self.device))
			tails_im = self.ent_im(torch.Tensor(tail).long().to(self.device))
			
			heads_re = self.ent_re(torch.Tensor(head).long().to(self.device))
			heads_im = self.ent_im(torch.Tensor(head).long().to(self.device))


			# Extract embeddings for all the tensors
			rels_re = self.rel_re(torch.arange(self.n_relations).long().to(self.device)).T
			rels_im = self.rel_im(torch.arange(self.n_relations).long().to(self.device)).T

			scores = torch.mm(tails_re*heads_re, rels_re)
			scores += torch.mm(tails_im*heads_im, rels_re)
			scores += torch.mm(tails_im*heads_im, rels_re)
			scores -= torch.mm(tails_re*heads_im, rels_im)


		return scores - self.tau