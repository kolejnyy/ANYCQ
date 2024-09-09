import torch
import json
import datetime
from time import time

import numpy as np
import networkx as nx

from tqdm import tqdm
from torch import nn

from argparse import ArgumentParser

from src.utils.metric_utils import evaluate_mrr
from src.model.model import ANYCQ
from src.utils.config_utils import dataset_from_config
from src.utils.data_utils import augment, baseline_nx
from src.predictor import *
from src.csp.cq_data import CQ_Data


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_qar(config_file, model_dir, model_name, n_pivots, exp_name, no_PE, only_neg, _start=None, _end=None):

	with open(config_file, 'r') as f:
		config = json.load(f)

	model = ANYCQ(None, config, use_PE= not no_PE)
	model = model.load_model(model_dir, name = model_name, use_PE = not no_PE)
	model.to(device)
	model.eval()

	dataset_config = config['test_data']
	dataset_name = dataset_config["folder_path"].split("/")[1]
	predictor_config = dataset_config["predictor"]

	with open(exp_name, 'w') as f:
		f.write(config_file + '\n')
		f.write(model_dir + '\n')
		f.write(model_name + '\n\n\n')
		f.write("No PE labels:  "+str(no_PE)+'\n')
		f.write("Only negative: "+str(only_neg)+'\n')

	data_files = [
		f"qar_test_{n_pivots}hub_efo1.json",
		f"qar_test_{n_pivots}hub_efo2.json",
		f"qar_test_{n_pivots}hub_efo3.json"
	]

	for val_id, data_file in enumerate(data_files):

		# Set the path to PE labels
		dataset_config['predictor']['PE_path'] = dataset_config['predictor']['PE_paths'][val_id] if 'PE_paths' in dataset_config['predictor'] else None 

		# Load the dataset
		dataset_config['data_file'] = data_file
		dataset_config['predictor']['temperature'] = dataset_config['predictor']['temperatures'][val_id]
		dataset = dataset_from_config(dataset_config, device)

		search_steps = 200

		with open(exp_name, 'a') as f:
			f.write(data_file + "  temp " + str(dataset_config['predictor']['temperature']) + '\n')

		total_acc = 0
		total_hit1 = 0

		easy_acc = 0
		easy_num = 0
		hard_acc = 0
		hard_num = 0
		fals_pos = 0

		num_pos = 0

		times = []

		start = 0 if _start is None else _start
		end = len(dataset) if _end is None else _end

		cq_types = list(dataset.data.keys())[start:end]

		for i, cq_type in tqdm(enumerate(cq_types), total=end-start):
			
			datum = dataset.data[cq_type][0]
			anss = datum[1][list(datum[1].keys())[0]]
			_type = datum[2]

			if _type == 'e':
				easy_num += 1
				num_pos += 1
				if only_neg:
					continue
			elif _type == 'h':
				hard_num += 1
				num_pos += 1
				if only_neg:
					continue 

			s_t = time()
			with torch.no_grad():
				out_data = model(
					dataset[start+i],
					search_steps,
					return_all_scores=True,
					return_all_assignments=True
				)
				times.append(time()-s_t)
				scores = out_data.all_cq_scores.cpu()[0]
				assgns = out_data.all_assignments.cpu()

			best_round = torch.argmax(scores)
			preds = assgns[out_data.pred_mask.cpu()][:, best_round]
			preds = [x.item() for x in preds]

			if _type == 'n':
				if scores[best_round] > 0:
					fals_pos += 1

			if preds in anss:
				total_hit1 += 1
				if scores[best_round] > 0:
					total_acc += 1
					if _type == 'e':
						easy_acc += 1
					else:
						hard_acc += 1

		precision = total_acc / (total_acc + fals_pos)  if total_acc > 0 else 0
		recall = total_acc / (num_pos)  if total_acc > 0 else 0
		f1_score = 2 / (1 / precision + 1 / recall) if total_acc > 0 else 0

		with open(exp_name, 'a') as f:
			f.write("False pos: "+str(fals_pos)+'\n')
			f.write("Precision: "+str(precision)+'\n')
			f.write("Recall:    "+str(recall)+'\n')
			f.write("F1 Score:  "+str(f1_score)+'\n')
			f.write("Easy rec:  "+str(easy_acc / easy_num)+'\n')
			f.write("Hard rec:  "+str(hard_acc / hard_num)+'\n')
			f.write("HITS@1:    "+str(total_hit1 / num_pos)+'\n')
			f.write(str(times)+'\n\n')


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument("--model_dir", type=str, help="Model directory")
	parser.add_argument("--model_name", type=str, help="Model name")
	parser.add_argument("--config_file", type=str, help="Config directory")
	parser.add_argument("--n_pivots", type=int, help="Split to evaluate")
	parser.add_argument("--exp_name", type=str, default='')
	parser.add_argument("--start", type=int, default=None, help="Id of the first sample")
	parser.add_argument("--end", type=int, default=None, help="Id of the last sample")
	parser.add_argument("--no_PE", action='store_true', default=False, help="Disable PE labels")
	parser.add_argument("--only_neg", action='store_true', default=False, help="Run only on negative queries")
	
	args = parser.parse_args()

	exp_name = "logs/"+args.exp_name+"".join(str(datetime.datetime.now()).split(' '))+'.out'
	

	test_qar(args.config_file, args.model_dir, args.model_name, args.n_pivots, exp_name, args.no_PE, args.only_neg, _start=args.start, _end=args.end)