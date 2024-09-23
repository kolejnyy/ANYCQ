import json
import duckdb
import pandas as pd
from time import time


dataset_name = 'NELL-EFO1'
split = 'valid'
gen_type = '3hub'

full_dict = {}
obs_dict = {}

for i in range(5):
	with open(f'data/{dataset_name}/qar_{split}_{gen_type}_efo3_{i}.json', 'r') as f:
		new_dict = json.load(f)
	full_dict = full_dict | new_dict

with open(f'data/{dataset_name}/qar_{split}_{gen_type}_full.json', 'r') as f:
	base_dict = json.load(f)

base_queries = list(base_dict.keys())

refined_dict = {}
processing_times = []

hard_count = 0

for cq_type in full_dict:

	grounding = full_dict[cq_type][0][0]
	
	grd, ans, p_time, id = (full_dict[cq_type][0])
	q_type = 'h' if len(base_dict[base_queries[id]][0][1]['f1'])==0 else 'e'
	if q_type == 'h':
		hard_count += 1
	
	processing_times.append(p_time)
	if len(ans['f1&f2&f3']) < 500:
		refined_dict[cq_type] = [[grd, ans, p_time, id, q_type]]

print("Refined ", len(refined_dict), 'queries')
print("Counted ", hard_count, 'hard queries')
print("Processing times:")
print("0  - 1   sec:  ", len([t for t in processing_times if t < 1]))
print("1  - 5   sec:  ", len([t for t in processing_times if t >= 1 and t < 5]))
print("5  - 10  sec:  ", len([t for t in processing_times if t >= 5 and t < 10]))
print("10 - 60  sec:  ", len([t for t in processing_times if t >= 10 and t < 60]))
print("60 - 200 sec:  ", len([t for t in processing_times if t >= 60 and t < 200]))
print("200+     sec:  ", len([t for t in processing_times if t >= 200]) )


with open(f'data/{dataset_name}/qar_{split}_{gen_type}_efo3_refined.json', 'w') as f:
	json.dump(refined_dict, f)