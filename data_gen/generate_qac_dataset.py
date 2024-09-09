import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time


dataset_name = 'FB15k-237-EFO1'
split = 'valid'
gen_type = '3piv'

kg_name = 'valid_kg.tsv' if split == 'test' else 'train_kg.tsv' 
kg_graph = pd.read_csv(f'data/{dataset_name}/{kg_name}', sep='\t', header = None)
kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

with open(f'data/{dataset_name}/vcq_{split}_{gen_type}_refined.json', 'r') as f:
	data = json.load(f)

qac_dataset = {}
potential_queries = []

for cq_type in data:
	
	grounding, easy_ans, hard_ans, eval_time = data[cq_type][0]
	answers = easy_ans['f1'] + hard_ans['f1']
	ps = np.ones(len(answers))
	ps[len(easy_ans['f1']):] = 1 if len(easy_ans['f1']) == 0 else 2*len(easy_ans['f1']) / len(hard_ans['f1'])
	ps /= np.sum(ps)

	entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
	relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]

	f1_relations = [rel for rel in relations if 'f1' in rel]
	rel_neg_values = pd.Index(kg_graph['a'].unique())

	for rel_id, a, b in f1_relations:
		if a == 'f1':
			rel_neg_values = rel_neg_values.intersection(kg_graph['a'][kg_graph['r']==grounding[rel_id]])
		if b == 'f1':
			rel_neg_values = rel_neg_values.intersection(kg_graph['b'][kg_graph['r']==grounding[rel_id]])

	rel_neg_values = [x for x in rel_neg_values if not x in answers]

	sample_num = min(5, min(len(rel_neg_values), len(answers)))
	if sample_num == 0:
		continue

	pos_grnds = [[x] for x in list(np.random.choice(answers, sample_num, replace = False, p = ps))]
	neg_grnds = [[x] for x in list(np.random.choice(rel_neg_values, sample_num, replace = False))]

	potential_queries.append((eval_time, cq_type, grounding, pos_grnds, neg_grnds))

potential_queries = sorted(potential_queries)

for _, cq_type, grounding, pos_grnds, neg_grnds in potential_queries[-300:]:
	qac_dataset[cq_type] = [[grounding, {'f1': pos_grnds}, {'f1': neg_grnds}]]

print(len(qac_dataset))

with open(f'data/{dataset_name}/qac_{split}_{gen_type}.json', 'w') as f:
	json.dump(qac_dataset, f, default=int)