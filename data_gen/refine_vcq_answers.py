import json
import duckdb
import pandas as pd
from time import time
from argparse import ArgumentParser
from multiprocessing import Process, Manager
from src.data.query_to_sql import query_to_sql


def run_with_limited_time(func, args, kwargs, _time):
	p = Process(target=func, args=args, kwargs=kwargs)
	p.start()
	p.join(_time)
	if p.is_alive():
		p.terminate()
		print("Timed out")

def process_query(cq_type, grounding, answers):
	s_t = time()
	sql_query = query_to_sql(cq_type, grounding)
	x = duckdb.sql(sql_query)
	query_ans = x.to_df()['f1'].tolist()
	answers.append([cq_type, grounding, query_ans, time()-s_t])
	print(query_ans)
	print("Time: ", time()-s_t)

def refine_answers(dataset, kg_name, split, gen_type, mod10, timeout):

	kg_graph = pd.read_csv(f'data/{dataset}/{kg_name}.tsv', sep='\t', header = None)
	kg_graph = kg_graph.rename(columns = {0: "a", 1: "r", 2: "b"})

	# DuckDB setup
	duckdb.sql("SET threads TO 16")
	duckdb.sql("SET memory_limit TO '124GB'")
	duckdb.sql("CREATE TABLE graph AS SELECT * FROM kg_graph")
	duckdb.sql("INSERT INTO graph SELECT * FROM kg_graph")

	# Load the VCQ data
	with open(f"data/{dataset}/vcq_{split}_{gen_type}hub.json", 'r') as f: 
		data = json.load(f)

	cq_types = list(data.keys())
	cq_types = [cq_types[i] for i in range(mod10, len(cq_types), 10)]

	with Manager() as manager:
		answers = manager.list()
		for cq_type in cq_types:
			grounding = data[cq_type][0][0]
			run_with_limited_time(process_query, (cq_type, grounding, answers), {}, timeout)
	
		answer_dict = {cq_type: [[grounding, ans, tim]] for cq_type, grounding, ans, tim in answers}

		with open(f'data/{dataset}/vcq_{split}_{gen_type}hub_{kg_name}_{mod10}.json', 'w') as f:
			json.dump(answer_dict, f)


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument("--dataset", type=str, help="The name of the dataset")
	parser.add_argument("--kg_name", type=str, help="The name of the KG graph (train / valid / test)")
	parser.add_argument("--split", type=str, help="The split to be evaluated (valid/test)")
	parser.add_argument("--n_pivots", type=int, help="Number of pivots in the generation split: 3, 4 or 5")
	parser.add_argument("--mod", type=int, help="i mod 10")
	parser.add_argument("--timeout", type=int, default = 60, help='Maximum number of seconds to process a query')
	args = parser.parse_args()

	refine_answers(args.dataset, args.kg_name, args.split, args.n_pivots, args.mod, args.timeout)