import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time


dataset_name = 'FB15k-237-EFO1'
split = 'test'

for gen_type in ['3hub', '4hub', '5hub']:
	for frees in ['efo1', 'efo2', 'efo3']:

		with open(f'data/{dataset_name}/qar_{split}_{gen_type}_{frees}.json', 'r') as f:
			data = json.load(f)

		cleaned_dict = {}

		i = 0

		for cq_type in data:

			row = data[cq_type]
			grounding = row[0][0]

			entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
			relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]

			filtered_relations = []
			for r, a, b in relations:
				if [r,a,b] in filtered_relations:
					print('Reeeemoving', r, a, b)
					continue
				git = True
				for r1, a1, b1 in filtered_relations:
					if a1 != b or b1 != a or r1 == r:
						continue
					# We have a=b', b=a' and r!= r'
					rel_id = grounding[r]
					rel_id1 = grounding[r1]
					if rel_id - (rel_id % 2) == rel_id1 - (rel_id1 % 2):
						git = False 
				if git:
					filtered_relations.append([r,a,b])
				else:
					print("Removing", r, a, b)
			
			new_cq_type = "&".join([('(' + r + '(' + a + ',' + b + '))') for r, a, b in filtered_relations])

			cleaned_dict[new_cq_type] = row


		print(len(cleaned_dict))

		with open(f'data/{dataset_name}/qar_{split}_{gen_type}_{frees}.json', 'w') as f:
			json.dump(cleaned_dict, f)