import json
import numpy as np
import networkx as nx
import pandas as pd
from time import time
from tqdm import tqdm
import torch
from torch import nn

from src.data.query_to_sql import query_to_sql
from argparse import ArgumentParser

from multiprocessing import Process, Manager
import duckdb

def run_with_limited_time(func, args, kwargs, _time):
	p = Process(target=func, args=args, kwargs=kwargs)
	p.start()
	p.join(_time)
	if p.is_alive():
		p.terminate()
		args[3].append(_time)
		# print("Timed out")

def process_query(cq_type, grounding, answers, times):
	s_t = time()
	sql_query = query_to_sql(cq_type, grounding) + ' LIMIT 1'
	x = duckdb.sql(sql_query)
	if len(x.to_df()) == 0:
		times.append(time()-s_t)
		return
	query_ans = list(x.to_df().iloc[0])
	answers.append(query_ans)
	times.append(time()-s_t)


def process_dataset(dataset, data_name, timeout):

	# Load the VCQ data
	with open(f"data/{dataset}-QAR/{data_name}", 'r') as f: 
		data = json.load(f)

	print(dataset, data_name)

	with Manager() as manager:
		total_acc = 0
		times = manager.list()
		easy_acc = 0; easy_tot = 0
		hard_acc = 0; hard_tot = 0
		for cq_type in tqdm(list(data.keys())):
			for grounding, corr_ans, q_type in (data[cq_type]):
				answers = manager.list()
				if q_type == 'e':
					easy_tot += 1
				else:
					hard_tot += 1
				run_with_limited_time(process_query, (cq_type, grounding, answers, times), {}, timeout)
				if len(answers) == 0:
					continue
				if answers[0] in corr_ans[list(corr_ans.keys())[0]]:
					total_acc += 1
					if q_type == 'e':
						easy_acc += 1
					else:
						hard_acc += 1
						
		print("Total: ", total_acc / len(data))
		print("Easy:  ", easy_acc / easy_tot)
		print("Hard:  ", hard_acc / hard_tot)
		print(times)



if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, help="Dataset name")
	parser.add_argument("--gen_type", type=str, help="3/4/5-piv")
	args = parser.parse_args()

	dataset = args.dataset
	gen_type = args.gen_type
	kg_name = 'valid_kg'
	timeout = 60


	kg_graph = pd.read_csv(f'data/{dataset}-QAR/{kg_name}.tsv', sep='\t', header = None)
	kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

	# DuckDB setup
	duckdb.sql("SET threads TO 16")
	duckdb.sql("SET memory_limit TO '124GB'")
	duckdb.sql("CREATE TABLE graph AS SELECT * FROM kg_graph")
	duckdb.sql("INSERT INTO graph SELECT * FROM kg_graph")

	for free_num in ['efo1', 'efo2', 'efo3']:
		data_name = f'qar_test_{gen_type}_{free_num}.json'
		process_dataset(dataset, data_name, timeout)