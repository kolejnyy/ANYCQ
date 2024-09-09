import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time


dataset_name = 'NELL-QAR'
gen_type = '5hub'
arity = 'efo1'
ent_num = 63361

with open(f'data/{dataset_name}/qar_test_{gen_type}_{arity}.json', 'r') as f:
	data = json.load(f)

cq_types = list(data.keys())

new_data = {}

anss_cal = 'f1'
if arity == 'efo2':
	anss_cal = 'f1&f2'
if arity == 'efo3':
	anss_cal = 'f1&f2&f3'

for cq_type in cq_types:
	entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
	relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]

	swap_ex = entities[0]
	neg_cq_type = cq_type
	neg_cq_type = neg_cq_type.replace('f1', 's0')
	neg_cq_type = neg_cq_type.replace(swap_ex+',', 'f1,')
	neg_cq_type = neg_cq_type.replace(swap_ex+')', 'f1)')

	# print(cq_type)
	# print(neg_cq_type)

	# print(data[cq_type])

	new_data[cq_type] = data[cq_type]
	anss = [x[0] for x in data[cq_type][0][1][anss_cal]]
	neg_ans = np.random.randint(0, ent_num)
	while neg_ans in anss:
		neg_ans = np.random.randint(0, ent_num)
	new_data[neg_cq_type] = [[data[cq_type][0][0] | {'s0': neg_ans}, {anss_cal: []}, 'n']]

	# print(new_data[neg_cq_type])

	# break

	
with open(f'data/{dataset_name}/qar_test_{gen_type}_{arity}_wneg.json', 'w') as f:
	print(len(new_data))
	json.dump(new_data, f)