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
from src.predictor.cqpred import SymCQPred
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


query = "((r1(s1,e1))&(!(r2(s2,e1))))&(r3(e1,f1))"

grounding = {
	"r1": 1,
	"r2": 10,
	"r3": 100,
	"s1": 2,
	"s2": 20,
}

print(query_to_sql(query, grounding))

# print(preds)

# train_kg = pd.read_csv('data/FB15k-237-QAR/train_kg.tsv', sep = '\t', header=None)
# valid_kg = pd.read_csv('data/FB15k-237-QAR/valid_kg.tsv', sep = '\t', header=None)

# diff_kg  = valid_kg[~valid_kg.apply(tuple,1).isin(train_kg.apply(tuple,1))].to_numpy()
# diff_kg  = torch.from_numpy(diff_kg.T).cuda()

# a = diff_kg[0]
# r = diff_kg[1]
# b = diff_kg[2]
# print(a.shape)
# print((predictor(a,r,b)>0.5).sum())
# print((sym_predictor(a,r,b)>0.5).sum())

# predictor_config['temperature'] = 20.0
# predictor = predictor_from_config(predictor_config['predictor'], 'cuda')

# heads = torch.randint(0, 14505, (5000,))
# rels = torch.randint(0, 474, (5000,))
# #rels = torch.arange(0, 474, 1)



# s_t = time()

# with torch.no_grad():
# 	tails = predictor.tail_scores(heads, rels)# * 20.0

# print(time()-s_t)
# print(torch.logsumexp(tails, 1))
# print(torch.nn.functional.softmax(tails, dim=1).max(axis=1))

# No augmentations

# Testing temperature 0.5
# 13
# Testing temperature 1.0
# 1764
# Testing temperature 2.0
# 5014
# Testing temperature 5.0
# 8458
# Testing temperature 20.0
# 11934

# Max
# Testing temperature 0.5
# 26
# Testing temperature 1.0
# 3386
# Testing temperature 2.0
# 9722
# Testing temperature 5.0
# 16442
# Testing temperature 20.0
# 23224