import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time


dataset_name = 'FB15k-237-EFO1'
split = 'valid'
gen_type = '3hub'

kg_name = 'valid_kg.tsv' if split == 'test' else 'train_kg.tsv' 
kg_graph = pd.read_csv(f'data/{dataset_name}/{kg_name}', sep='\t', header = None)
kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

with open(f'data/{dataset_name}/vcq_leftovers/vcq_{split}_{gen_type}_refined.json', 'r') as f:
	data = json.load(f)

easy_qar_dataset = {}
hard_qar_dataset = {}

for cq_type in data:
	
	grounding, easy_ans, hard_ans, _ = data[cq_type][0]
	
	if len(easy_ans['f1']) == 0:
		hard_qar_dataset[cq_type] = [[grounding, easy_ans, hard_ans, 'h']]
	elif len(hard_ans['f1']) > 0:
		easy_qar_dataset[cq_type] = [[grounding, easy_ans, hard_ans, 'e']]

with open(f'data/{dataset_name}/qar_{split}_{gen_type}_efo1_new.json', 'w') as f:
	json.dump(easy_qar_dataset | hard_qar_dataset, f, default=int)

print("All:         ", len(easy_qar_dataset | hard_qar_dataset))
print("Trivial:     ", len(easy_qar_dataset))
print("Non-trivial: ", len(hard_qar_dataset))