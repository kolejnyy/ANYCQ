import json
import duckdb
import pandas as pd
from time import time


dataset_name = 'FB15k-237-EFO1'
split = 'valid'
gen_type = '3hub'

full_dict = {}
obs_dict = {}

for i in range(10):
	with open(f'data/{dataset_name}/vcq_{split}_{gen_type}_valid_kg_{i}.json', 'r') as f:
		new_dict = json.load(f)
	full_dict = full_dict | new_dict

for i in range(10):
	with open(f'data/{dataset_name}/vcq_{split}_{gen_type}_train_kg_{i}.json', 'r') as f:
		new_dict = json.load(f)
	obs_dict = obs_dict | new_dict

refined_dict = {}
processing_times = []

for cq_type in full_dict:
	
	if not cq_type in obs_dict.keys():
		print('Type not found')
		continue
	if full_dict[cq_type][0][0] != obs_dict[cq_type][0][0]:
		print('Unmatching groundings!')
		continue

	grounding = full_dict[cq_type][0][0]
	
	easy_ans = obs_dict[cq_type][0][1]
	# easy_ans = full_dict[cq_type][0][1]
	hard_ans = [x for x in full_dict[cq_type][0][1] if not x in obs_dict[cq_type][0][1]]
	if len(hard_ans) == 0 or len(easy_ans) + len(hard_ans) > 500:
		continue
	# if len(easy_ans) > 500:
	# 	continue

	refined_dict[cq_type] = [[grounding, {'f1': easy_ans}, {'f1': hard_ans}, obs_dict[cq_type][0][2]]]
	# refined_dict[cq_type] = [[grounding, easy_ans, full_dict[cq_type][0][2], full_dict[cq_type][0][3]]]

	processing_times.append(full_dict[cq_type][0][2])

print("Refined ", len(refined_dict), 'queries')
print("Processing times:")
print("0  - 1   sec:  ", len([t for t in processing_times if t < 1]))
print("1  - 5   sec:  ", len([t for t in processing_times if t >= 1 and t < 5]))
print("5  - 10  sec:  ", len([t for t in processing_times if t >= 5 and t < 10]))
print("10 - 60  sec:  ", len([t for t in processing_times if t >= 10 and t < 60]))
print("60 - 200 sec:  ", len([t for t in processing_times if t >= 60 and t < 200]))
print("200+     sec:  ", len([t for t in processing_times if t >= 200]) )


with open(f'data/{dataset_name}/vcq_{split}_{gen_type}_refined.json', 'w') as f:
	json.dump(refined_dict, f)