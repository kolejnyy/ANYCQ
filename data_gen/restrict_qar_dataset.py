import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time


dataset_name = 'FB15k-237-EFO1'
split = 'test'
gen_type = '5hub'
fnum = 'efo3'

kg_name = 'valid_kg.tsv' if split == 'test' else 'train_kg.tsv' 
kg_graph = pd.read_csv(f'data/{dataset_name}/{kg_name}', sep='\t', header = None)
kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

# with open(f'data/{dataset_name}/leftovers/qar_{split}_{gen_type}_full.json', 'r') as f:
# 	data = json.load(f)
with open(f'data/{dataset_name}/vcq_leftovers/qar_{split}_{gen_type}_{fnum}_full.json', 'r') as f:
	full_data = json.load(f)
	
with open(f'data/{dataset_name}/vcq_leftovers/qar_{split}_{gen_type}_full.json', 'r') as f:
	base_data = json.load(f)

new_dataset = {}
queries = []

for cq_type in full_data:
	grounding, easy_ans, ex_time, q_id = full_data[cq_type][0]
	queries.append([ex_time, cq_type, grounding, easy_ans, q_id])

type_list = list(base_data.keys())

queries = sorted(queries)[-400:]

print(len(queries))
for _, cq_type, grounding, easy_ans, q_id in queries:
	anss = easy_ans['f1&f2&f3']
	# anss = [[x] for x in anss]
	new_dataset[cq_type] = [[grounding, {'f1&f2&f3' : anss}, 'h' if len(base_data[type_list[q_id]][0][1]['f1']) == 0 else 'e']]

with open(f'data/{dataset_name[:-5]}-QAR/qar_{split}_{gen_type}_{fnum}_new.json', 'w') as f:
	json.dump(new_dataset, f, default=int)

