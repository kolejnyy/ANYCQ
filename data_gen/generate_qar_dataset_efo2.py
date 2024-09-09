import re
import json
from copy import copy
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

from argparse import ArgumentParser
from multiprocessing import Process, Manager

import duckdb

from src.data.query_to_sql import query_to_sql

def run_with_limited_time(func, args, kwargs, _time):
	p = Process(target=func, args=args, kwargs=kwargs)
	p.start()
	p.join(_time)
	if p.is_alive():
		print("Time out!")
		p.terminate()

def process_query(cq_type, grounding, answers, anss, query_id):
	s_t = time()
	sql_query = query_to_sql(cq_type, grounding)
	sql_query += ' AND (f1.a IN ('+', '.join([str(x) for x in anss])+'))'
	
	x = duckdb.sql(sql_query)
	query_ans = x.to_df()

	answers.append([cq_type, grounding, query_ans, time()-s_t, query_id])



def generate_efo2_data(dataset_name, kg_name, split, gen_type, mod, timeout):

	kg_graph = pd.read_csv(f'data/{dataset_name}/{kg_name}', sep='\t', header = None)
	kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

	duckdb.sql("SET threads TO 16")
	duckdb.sql("SET memory_limit TO '124GB'")
	duckdb.sql("CREATE TABLE graph AS SELECT * FROM kg_graph")
	duckdb.sql("INSERT INTO graph SELECT * FROM kg_graph")

	with open(f'data/{dataset_name}/qar_{split}_{gen_type}_full.json', 'r') as f:
		data = json.load(f)

	easy_qar_dataset = {}
	hard_qar_dataset = {}

	query_id = mod

	cq_types = list(data.keys())
	cq_types = [cq_types[i] for i in range(mod, len(cq_types), 10)]

	with Manager() as manager:
		answers = manager.list()

		for cq_type in tqdm(cq_types):
			
			grounding, easy_ans, hard_ans = data[cq_type][0]
			anss = easy_ans['f1'] + hard_ans['f1']
			
			entities = np.unique([x for x in re.split('[^a-zA-Z0-9]', cq_type) if (len(x)>0 and x[0]!='r')])
			relations = [[y for y in re.split('[^a-zA-Z0-9!]', x) if len(y)>0] for x in cq_type.split('&') if len(x)>0]
			ex_vars = [x for x in entities if x[0] == 'e']

			free_vars = np.random.choice(ex_vars, 2, replace = False)
			
			for var in free_vars:
				new_cq_type = cq_type.replace(var+',', 'f2,')
				new_cq_type = new_cq_type.replace(var+')', 'f2)')
				new_grounding = copy(grounding)
				new_grounding.pop(var, None)
				run_with_limited_time(process_query, (new_cq_type, new_grounding, answers, anss, query_id), {}, timeout)
				
			query_id += 10
		
		efo2_dataset = {}
		for cq_type, grounding, res_df, time, q_id in answers:
			results = res_df.to_numpy()
			results = [list(row) for row in results]
			if len(results) < 500:
				efo2_dataset[cq_type] = [[grounding, {"f1&f2" : results}, time, q_id]]

		with open(f'data/{dataset_name}/qar_{split}_{gen_type}_efo2_{kg_name[:-4]}_{mod}.json', 'w') as f:
			json.dump(efo2_dataset, f, default = int)


if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, help="The name of the dataset")
	parser.add_argument("--kg_name", type=str, help="The name of the KG graph (train / valid / test)")
	parser.add_argument("--split", type=str, help="The split to be evaluated (valid/test)")
	parser.add_argument("--n_pivots", type=int, help="Number of pivots in the generation split: 3, 4 or 5")
	parser.add_argument("--mod", type=int, help="i mod 10")
	parser.add_argument("--timeout", type=int, default = 60, help='Maximum number of seconds to process a query')
	args = parser.parse_args()

	generate_efo2_data(args.dataset, args.kg_name+'.tsv', args.split, str(args.n_pivots)+'hub', args.mod, args.timeout)