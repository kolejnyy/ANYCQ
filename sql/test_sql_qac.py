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

def process_query(cq_type, grounding, x, answers, times):
	s_t = time()
	sql_query = query_to_sql(cq_type, grounding) + f'AND f1.a={x} LIMIT 1'
	x = duckdb.sql(sql_query)
	if len(x.to_df()) == 0:
		times.append(time()-s_t)
		return
	query_ans = list(x.to_df().iloc[0])
	answers.append(query_ans)
	times.append(time()-s_t)


def process_dataset(dataset, data_name, timeout):

	# Load the VCQ data
	with open(f"data/{dataset}-QAC/{data_name}", 'r') as f: 
		data = json.load(f)

	print(dataset, data_name)

	with Manager() as manager:

		tot_recall = 0
		tot_prec = 0
		tot_f1 = 0
		n_samples = 0

		times = manager.list()
		for cq_type in list(data.keys()) if len(data) < 10 else tqdm(list(data.keys())[:10]):
			for grounding, pos_anss, neg_anss in tqdm(data[cq_type]) if len(data[cq_type]) > 10 else data[cq_type]:
				
				n_samples += 1

				tp = 0; fp = 0
				tn = 0; fn = 0

				for x in pos_anss['f1']:
					answers = manager.list()
					run_with_limited_time(process_query, (cq_type, grounding, x[0], answers, times), {}, timeout)
					if len(answers) == 0:
						fn += 1
					else:
						tp += 1
				
				for y in neg_anss['f1']:
					answers = manager.list()
					run_with_limited_time(process_query, (cq_type, grounding, y[0], answers, times), {}, timeout)
					if len(answers) == 0:
						tn += 1
					else:
						fp += 1

				_recall = tp / max(1, tp + fn)
				_precision = tp / max(1, tp + fp)
				_f1_score = 0
				if tp != 0:
					_f1_score = 2 / (1/_recall + 1/_precision)
				
				tot_recall += _recall
				tot_prec += _precision
				tot_f1 += _f1_score

		print("Recall:     ", tot_recall / n_samples)
		print("Precision:  ", tot_prec / n_samples)
		print("F1-Score:   ", tot_f1 / n_samples)
		# print(times)



if __name__ == "__main__":

	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, help="Dataset name")
	args = parser.parse_args()

	dataset = args.dataset
	kg_name = 'valid_kg'
	timeout = 60


	kg_graph = pd.read_csv(f'data/{dataset}-QAC/{kg_name}.tsv', sep='\t', header = None)
	kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

	# DuckDB setup
	duckdb.sql("SET threads TO 16")
	duckdb.sql("SET memory_limit TO '124GB'")
	duckdb.sql("CREATE TABLE graph AS SELECT * FROM kg_graph")
	duckdb.sql("INSERT INTO graph SELECT * FROM kg_graph")

	data_names = [
		# "qac_test_type0001.json",
		# "qac_test_type0002.json",
		# "qac_test_type0005.json",
		# "qac_test_type0006.json",
		"qac_test_type0009.json",
		"qac_test_type0010.json",
		# "qac_test_3piv.json"
		# "qac_test_4piv.json"
		# "qac_test_5piv.json"
	]

	for data_name in data_names:
		process_dataset(dataset, data_name, timeout)