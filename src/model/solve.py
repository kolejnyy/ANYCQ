import re
import torch
import numpy as np
from tqdm import tqdm
from src.csp.cq_data import CQ_Data



def solve_global(model, dataset, idx, n_steps):
	
	with torch.no_grad():
		out_data = model(dataset[idx],
						n_steps,
						return_all_scores=True,
						return_all_assignments = True
		)

	scores = out_data.all_cq_scores.cpu()
	assgns = out_data.all_assignments.cpu()

	max_score = scores.max()
	sorted_assgns = assgns[:,torch.argsort(scores, descending=True)]
	pred_id = torch.arange(assgns.shape[0])[out_data.pred_mask]
	answers = torch.unique_consecutive(sorted_assgns[pred_id])

	return answers, max_score


def get_cwa_values(dataset, closed_data, query_type, index):

	res_mask = torch.zeros(dataset.n_entities).long()

	for qc_type in query_type.split('|'):
		
		# Find the relations that mention the predicted variable
		relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in qc_type.split('&') if len(x)>0]
		relations = [rel for rel in relations if 'f1' in rel]
		relations = [rel if len(rel)==3 else [rel[0]+rel[1], rel[2], rel[3]] for rel in relations]
		
		# Get the input assignments to relations and constant nodes
		assignments = dataset.data[query_type][index][0] 
		valid_mask = torch.ones(dataset.n_entities).long()

		# Mask all head/tail restrictions
		for rel_id, head, tail in relations:
			if rel_id[0] == '!':
				continue
			if head=='f1':
				new_mask = torch.zeros(dataset.n_entities).long()
				new_mask[closed_data[0][closed_data[1]==assignments[rel_id]]] = 1
				valid_mask *= new_mask
			else:
				new_mask = torch.zeros(dataset.n_entities).long()
				new_mask[closed_data[2][closed_data[1]==assignments[rel_id]]] = 1
				valid_mask *= new_mask

		res_mask += valid_mask

	return torch.arange(dataset.n_entities)[res_mask>0]



def solve_marginal(model, dataset, closed_data, idx, n_steps, gpu_cap = 900000):

	# Find the type of the query and queits index
	qc_type = np.argmax(dataset.cq_off>idx)-1
	index = idx - dataset.cq_off[qc_type]
	qc_type = dataset.cq_types[qc_type]

	# Generate the set of possible values for the free variable under CWA
	val_set = get_cwa_values(dataset, closed_data, qc_type, index)
	if len(val_set)==0:
		val_set = torch.arange(len(dataset))
	step = gpu_cap//dataset.get_grounded(idx,0).num_val
	
	# If there are two many possible values, sample ones for 2 sets of evaluation
	if len(val_set)>1000//n_steps*step:
		rand_perm = torch.randperm(len(val_set))
		val_set = val_set[rand_perm[:1000//n_steps*step]]

	scores = torch.zeros(len(val_set))

	for i in (range(0, len(val_set), step)):
		j = min(len(val_set), i+step)
		input_data = CQ_Data.collate([dataset.get_grounded(idx, val_set[k]) for k in range(i,j)])
		with torch.no_grad():
			out_data = model(input_data,
							n_steps,
							return_all_scores=True
			)
		scores[i:j] = torch.max(out_data.all_cq_scores.cpu(), dim=1).values

	answers = val_set[torch.argsort(scores, descending=True)]
	
	return answers, scores.max()