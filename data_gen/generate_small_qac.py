import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time

from src.predictor import *
from src.utils.data_utils import augment, baseline_nx


dataset_name = 'NELL-EFO1'
split = 'test'

kg_name = 'valid_kg.tsv' if split == 'test' else 'train_kg.tsv' 
kg_graph = pd.read_csv(f'data/{dataset_name}/{kg_name}', sep='\t', header = None)
kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

reasoner_perfect =  MatrixReasoner(
	'models/'+dataset_name+'/reasoner_perfect_test/qto',
	63361,
	400,
	'cuda'
)


for gen_type in ['type0001', 'type0002', 'type0005', 'type0006', 'type0009', 'type0010']:

	with open(f'data/{dataset_name}/standard/{split}_{gen_type}_real_EFO1_qaa.json', 'r') as f:
		data = json.load(f)

	qac_dataset = {}
	potential_queries = []
	cq_type = list(data.keys())[0]

	for i in range(len(data[cq_type])):
		
		grounding, easy_ans, hard_ans = data[cq_type][i]
		
		graph = augment(baseline_nx(cq_type), grounding).reverse()
		free_node = -1
		for node_id, node_info in graph.nodes(data=True):
			if node_info['name']=='f1':
				free_node = node_id
				break
		result_perf = reasoner_perfect.process(graph, free_node, 'max')
		corr_answers = [x.item() for x in list(result_perf.nonzero().flatten())]

		answers = [x[0] for x in easy_ans['f1'] + hard_ans['f1'] if x[0] in corr_answers]
		ps = np.ones(len(answers))
		ps[len(easy_ans['f1']):] = 1 if len(easy_ans['f1']) == 0 else 2*len(easy_ans['f1']) / len(hard_ans['f1'])
		ps /= np.sum(ps)

		entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
		relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]

		f1_relations = [rel for rel in relations if 'f1' in rel and len(rel)==3]
		rel_neg_values = pd.Index(kg_graph['a'].unique())

		for rel_id, a, b in f1_relations:
			if a == 'f1':
				rel_neg_values = rel_neg_values.intersection(kg_graph['a'][kg_graph['r']==grounding[rel_id]])
			if b == 'f1':
				rel_neg_values = rel_neg_values.intersection(kg_graph['b'][kg_graph['r']==grounding[rel_id]])

		rel_neg_values = [x for x in rel_neg_values if not x in answers]

		sample_num = min(10, min(len(rel_neg_values), len(answers)))
		if sample_num < 5:
			continue

		pos_grnds = [[x] for x in list(np.random.choice(answers, sample_num, replace = False, p = ps))]
		neg_grnds = [[x] for x in list(np.random.choice(rel_neg_values, sample_num, replace = False))]

		potential_queries.append((cq_type, grounding, pos_grnds, neg_grnds))
		if len(potential_queries) > 500:
			break

	qac_dataset[cq_type] = []
	for cq_type, grounding, pos_grnds, neg_grnds in potential_queries[:500]:
		qac_dataset[cq_type].append([grounding, {'f1': pos_grnds}, {'f1': neg_grnds}])

	print(gen_type, ":  len =", len(qac_dataset[cq_type]))

	with open(f'data/{dataset_name}/qac_{split}_{gen_type}.json', 'w') as f:
		json.dump(qac_dataset, f, default=int)