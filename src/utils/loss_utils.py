import torch


def get_distributions(dataset):

	res = [None, None, None]
	for i in range(3):
		ents, ent_counts = torch.unique(dataset[i], return_counts=True)
		ent_counts = ent_counts/ent_counts.sum()
		res[i] = torch.zeros(len(ents))
		res[i][ents] = ent_counts
	
	return res[0], res[1], res[2]


def evaluate_regularisation(predictor, heads, rels, tails, dist_h, dist_r, dist_t):

	# Entity embeddings
	heads_re = (predictor.ent_re(heads).T ** 2)
	heads_im = (predictor.ent_im(heads).T ** 2)
	heads_abs = torch.sqrt(heads_re + heads_im)
	reg_score = torch.sum((heads_abs ** 3))
	
	# Relation embeddings
	rels_re = (predictor.rel_re(rels).T ** 2)
	rels_im = (predictor.rel_im(rels).T ** 2)
	rels_abs = torch.sqrt(rels_re + rels_im)
	reg_score += torch.sum((rels_abs ** 3))

	# Tail embeddings
	tails_re = (predictor.ent_re(tails).T ** 2)
	tails_im = (predictor.ent_im(tails).T ** 2)
	tails_abs = torch.sqrt(tails_re + tails_im)
	reg_score += torch.sum((tails_abs ** 3))

	return reg_score / heads.shape[0]



def evaluate_n3_loss(predictor, heads, rels, tails, dist_h, dist_r, dist_t):

	# Define helper functions
	softmax = torch.nn.Softmax(dim=1)
	criterion = torch.nn.CrossEntropyLoss()

	rev_rels = rels - 2*(rels%2) + 1

	# Evaluate the head loss
	L_h = (criterion((predictor.head_scores(tails, rels, train=True)), heads)
		+ criterion((predictor.tail_scores(tails, rev_rels, train=True)), heads)) / 2
	# Evaluate the rel loss
	L_r = (criterion((predictor.rel_scores(heads, tails, train=True)), rels)
		+ criterion((predictor.rel_scores(tails, heads, train=True)), rev_rels) ) / 2
	# Evaluate the tail loss
	L_t = (criterion((predictor.tail_scores(heads, rels, train=True)), tails)
		+ criterion((predictor.head_scores(heads, rev_rels, train=True)), tails)) / 2

	# Evaluate the regularisation loss
	L_reg = evaluate_regularisation(predictor, heads, rels, tails, dist_h, dist_r, dist_t)

	return L_h, L_r, L_t, L_reg
