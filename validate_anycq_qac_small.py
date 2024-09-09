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


def test_qac(config_file, model_dir, model_name, exp_name, _start=None, _end=None):

	with open(config_file, 'r') as f:
		config = json.load(f)

	model = ANYCQ(None, config)
	model = model.load_model(model_dir, name = model_name)
	model.to(device)
	model.eval()

	dataset_config = config['val_data']
	dataset_name = dataset_config["folder_path"].split("/")[1]
	predictor_config = dataset_config["predictor"]

	with open(exp_name, 'w') as f:
		f.write(config_file + '\n')
		f.write(model_dir + '\n')
		f.write(model_name + '\n\n\n')

	for val_id, data_file in enumerate(dataset_config['data_files']):

		# Set the path to PE labels
		dataset_config['predictor']['PE_path'] = dataset_config['predictor']['PE_paths'][val_id] if 'PE_paths' in dataset_config['predictor'] else None 

		# Load the dataset
		dataset_config['data_file'] = data_file
		dataset = dataset_from_config(dataset_config, device)
		cq_type = dataset.cq_types[0]

		search_steps = 30

		with open(exp_name, 'a') as f:
			f.write(cq_type + '\n')

		total_f1 = 0

		start = 0 if _start is None else _start
		end = len(dataset) if _end is None else _end

		for i, datum in tqdm(enumerate(dataset.data[cq_type][start:end]), total=end-start):

			in_pos_data = CQ_Data.collate([dataset.get_grounded(start + i, pot_s[0]) for pot_s in datum[1]['f1']])
			in_neg_data = CQ_Data.collate([dataset.get_grounded(start + i, pot_s[0]) for pot_s in datum[2]['f1']])

			# Evaluate predictions for the positive answers
			with torch.no_grad():
				out_data = model(
						in_pos_data,
						search_steps,
						return_all_scores=True,
				)
			tp = (torch.max(out_data.all_cq_scores.cpu(), dim=1)[0] > 0).sum().item()
			fn = len(datum[1]['f1']) - tp

			if tp == 0:
				continue

			# Evaluate predictions for the negative answers
			with torch.no_grad():
				out_data = model(
						in_neg_data,
						search_steps,
						return_all_scores=True,
				)
			fp = (torch.max(out_data.all_cq_scores.cpu(), dim=1)[0] > 0).sum().item()

			prec = tp / (tp + fp)
			rec = tp / (tp + fn)
			total_f1 += 2 / (1/prec + 1/rec)

		with open(exp_name, 'a') as f:
			f.write("F1-score:  "+str(total_f1 / (end-start))+'\n')


if __name__ == '__main__':

	parser = ArgumentParser()
	parser.add_argument("--model_dir", type=str, help="Model directory")
	parser.add_argument("--model_name", type=str, help="Model name")
	parser.add_argument("--config_file", type=str, help="Config directory")
	parser.add_argument("--exp_name", type=str, default='')
	parser.add_argument("--start", type=int, default=None, help="Id of the first sample")
	parser.add_argument("--end", type=int, default=None, help="Id of the last sample")
	args = parser.parse_args()

	exp_name = "logs/"+args.exp_name+"".join(str(datetime.datetime.now()).split(' '))+'.out'
	

	test_qac(args.config_file, args.model_dir, args.model_name, exp_name, _start=args.start, _end=args.end)