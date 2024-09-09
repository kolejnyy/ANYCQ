import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time


dataset_name = 'NELL-EFO1'
split = 'test'
gen_type = '5hub'

kg_name = 'valid_kg.tsv' if split == 'test' else 'train_kg.tsv' 
kg_graph = pd.read_csv(f'data/{dataset_name}/{kg_name}', sep='\t', header = None)
kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

# with open(f'data/{dataset_name}/leftovers/qar_{split}_{gen_type}_full.json', 'r') as f:
# 	data = json.load(f)
with open(f'data/{dataset_name}/vcq_{split}_{gen_type}_refined.json', 'r') as f:
	base_data = json.load(f)

new_dataset = {}
queries = []

for cq_type in base_data:
	
	grounding, easy_ans, hard_ans, ex_time = base_data[cq_type][0]
	queries.append([ex_time, cq_type, grounding, easy_ans, hard_ans])


queries = sorted(queries)[-400:]

for _, cq_type, grounding, easy_ans, hard_ans in queries:
	anss = easy_ans['f1'] + hard_ans['f1']
	anss = [[x] for x in anss]
	new_dataset[cq_type] = [[grounding, {'f1' : anss}, 'h' if len(easy_ans['f1'])==0 else 'e']]

with open(f'data/{dataset_name[:-5]}-QAR/qar_{split}_{gen_type}_efo1_new.json', 'w') as f:
	json.dump(new_dataset, f, default=int)

print(len(new_dataset))