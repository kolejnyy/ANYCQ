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


def validate_qar(config_file, model_dir, model_name, exp_name, _start=None, _end=None):

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
		dataset_config['predictor']['temperature'] = dataset_config['predictor']['temperatures'][val_id]
		dataset = dataset_from_config(dataset_config, device)

		search_steps = 200

		with open(exp_name, 'a') as f:
			f.write(data_file + "  temp " + str(dataset_config['predictor']['temperature']) + '\n')

		total_acc = 0
		total_hit1 = 0

		easy_acc = 0
		hard_acc = 0
		easy_num = 0
		hard_num = 0

		start = 0 if _start is None else _start
		end = len(dataset) if _end is None else _end

		cq_types = list(dataset.data.keys())[start:end]

		for i, cq_type in tqdm(enumerate(cq_types), total=end-start):
			
			datum = dataset.data[cq_type][0]
			anss = datum[1]['f1']
			_type = datum[2]

			if _type == 'e':
				easy_num += 1
			else:
				hard_num += 1

			with torch.no_grad():
				out_data = model(
					dataset[start+i],
					search_steps,
					return_all_scores=True,
					return_all_assignments=True
				)
				scores = out_data.all_cq_scores.cpu()[0]
				assgns = out_data.all_assignments.cpu()

			preds = assgns[out_data.pred_mask.cpu()][0]
			best_round = torch.argmax(scores)

			if [preds[best_round].item()] in anss:
				total_hit1 += 1
				if scores[best_round] > 0:
					total_acc += 1
					if _type == 'e':
						easy_acc += 1
					else:
						hard_acc += 1


		with open(exp_name, 'a') as f:
			f.write("Accuracy:  "+str(total_acc / (end-start))+'\n')
			f.write("Easy acc:  "+str(easy_acc / max(1, easy_num))+'\n')
			f.write("Hard acc:  "+str(hard_acc / max(1, hard_num))+'\n')
			f.write("HITS@1:    "+str(total_hit1 / (end-start))+'\n')


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
	

	validate_qar(args.config_file, args.model_dir, args.model_name, exp_name, _start=args.start, _end=args.end)