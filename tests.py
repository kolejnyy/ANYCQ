import torch
import json
import ijson
import datetime
import re
import os
from time import time

import pandas as pd
import numpy as np
import networkx as nx

import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm
import torch
from torch import nn
from torch_scatter import scatter_max
from torch.utils.data import DataLoader


from argparse import ArgumentParser

from torch.utils.data import DataLoader

from src.utils.metric_utils import evaluate_mrr
from src.csp.cq_data import CQ_Data
from src.model.model import ANYCQ
from src.data.dataset import CQA_Dataset
from src.utils.config_utils import dataset_from_config
from src.utils.data_utils import augment, baseline_nx


from src.predictor import *
from src.data.dataset import CQA_Dataset
from src.csp.cq_data import CQ_Data
from src.utils.config_utils import dataset_from_config, predictor_from_config
from src.utils.metric_utils import evaluate_mrr, validate_predictor_mrr, validate_predictor_accuracy, evaluate_class_metrics
from src.model.model import ANYCQ
from src.model.loss import reward_improve, reinforce_loss
from src.data.vcq_gen import generate_large_query_dataset
from src.data.query_to_sql import query_to_sql
from src.model.solve import solve_global, solve_marginal, get_cwa_values

# import duckdb
# from multiprocessing import Process, Manager




with open('configs/model/model_FB15k-237_qar.json', 'r') as f:
	config = json.load(f)

predictor_config = config['val_data']['predictor']['predictor']
complEx = predictor_from_config(predictor_config, 'cuda:1')
sym_predictor = predictor_from_config(config['val_data']['predictor'], 'cuda:1')

sym_dataset = dataset_from_config(config['val_data'], 'cuda:1')
print(config['val_data']['predictor']['type'])
anycq = ANYCQ(None, config)
anycq = anycq.load_model('models/FB15k-237-EFO1/anycq', '_checkpoint_350000')
anycq.to('cuda:1')

cq_types = list(sym_dataset.data.keys())

for i in range(10):

	s_t = time()

	sym_datum = sym_dataset.data[cq_types[i]][0]
	anss = sym_datum[1][list(sym_datum[1].keys())[0]]

	with torch.no_grad():

		out_data = anycq(
			sym_dataset[i],
			150,
			return_all_scores=True,
			return_all_assignments=True
		)
		scores = out_data.all_cq_scores.cpu()[0]
		assgns = out_data.all_assignments.cpu()

	print(time()-s_t)

	best_round = torch.argmax(scores)
	preds = assgns[out_data.pred_mask.cpu()][:, best_round]
	preds = [x.item() for x in preds]

	if preds in anss:
		print(f"Sample {i}:", scores.max(), 'correct')
	else:
		print(f"Sample {i}:", scores.max(), 'wrong')
		if scores.max() == 1:
			print(sym_datum[0])
			print(assgns[:, best_round])
# dataset_name = 'FB15k-237-EFO1'
# split = '4piv'
# fnum = 'efo2'
# n_samples = 400


# P_head = torch.zeros((474, 14505)).cuda()
# P_tail = torch.zeros((474, 14505)).cuda()

# for i in tqdm(range(474)):
	
# 	x = torch.load(f'models/FB15k-237-EFO1/reasoner_test_20.0/fit/P_{i}.pt').cuda()
# 	y = torch.load(f'models/FB15k-237-EFO1/reasoner_perfect/fit/P_{i}.pt').cuda()
# 	x = (x+y).to_dense()
# 	# x[x<0.5] = 0
# 	# x[x>0.2] = 1
# 	poss_head = torch.max(x, dim=1)[0]
# 	poss_tail = torch.max(x, dim=0)[0]
# 	P_head[i, poss_head>0.5] = 1
# 	P_tail[i, poss_tail>0.5] = 1

# torch.save(P_head.to_sparse(), 'models/FB15k-237-EFO1/reasoner_test_20.0/fit/PE_head.pt')
# torch.save(P_tail.to_sparse(), 'models/FB15k-237-EFO1/reasoner_test_20.0/fit/PE_tail.pt')

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# data_path = 'data/FB15k-237-EFO1'

# with open('configs/model/model_FB15k-237_perfect.json', 'r') as f:
# 	model_config = json.load(f)
# predictor = predictor_from_config(model_config['test_data']['predictor'], device)
# model = ANYCQ(None, model_config)
# model = model.load_model('models/FB15k-237-EFO1/anycq', '_checkpoint_350000')
# model.to(device)

# print(predictor)


# generate_large_query_dataset('data_path', dataset_name, dataset_size, n_pivots, min_desired_vertex_num, p_out)