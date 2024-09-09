import json
import numpy as np
import networkx as nx

from tqdm import tqdm
import torch
from torch import nn
from src.utils.config_utils import dataset_from_config
from src.utils.data_utils import augment, baseline_nx
from src.predictor import *



device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_reasoner(config_path):

	with open(config_path, 'r') as f:
		reasoner_config = json.load(f)

	dataset_name = reasoner_config['dataset']

	with open('configs/datasets/dataset_test_'+dataset_name+'.json', 'r') as f:
		data_config = json.load(f)
	reasoner_type = reasoner_config['predictor']['scaling_rule']

	for type_id, data_file in enumerate(data_config['data_files']):

		temperature = str(reasoner_config['test_temperatures'][type_id])

		reasoner = MatrixReasoner(
			'models/'+dataset_name+'/reasoner_test_'+temperature+'/'+reasoner_type,
			data_config['n_entities'],
			data_config['n_relations'],
			device
		)

		# Load the dataset
		data_config['data_file'] = data_file 
		dataset = dataset_from_config(data_config, device)
		cq_type = dataset.cq_types[0]

		print(cq_type, "using t=", temperature)

		total_f1 = 0

		for datum in (dataset.data[cq_type][:50]):

			graph = augment(baseline_nx(cq_type), datum[0]).reverse()
			free_node = -1
			for node_id, node_info in graph.nodes(data=True):
				if node_info['name']=='f1':
					free_node = node_id
					break

			result = reasoner.process(graph, free_node, 'max' if reasoner_type == 'fit' else 'prod')
			
			pos_anss = torch.Tensor(datum[1]['f1']).flatten().long()
			neg_anss = torch.Tensor(datum[2]['f1']).flatten().long()
			
			tp = (result[pos_anss] >= 0.5).sum()
			fp = (result[neg_anss] >= 0.5).sum()
			fn = (result[pos_anss] < 0.5).sum()
			tn = (result[neg_anss] < 0.5).sum()

			if tp == 0:
				continue

			prec = tp / (tp + fp)
			rec = tp / (tp + fn)
			total_f1 += 2 / (1 / prec + 1 / rec)

		print('Average F1: ', total_f1.item() / 50)


config_paths = [
	# 'configs/reasoners/fit_fb15k-237.json',
	# 'configs/reasoners/qto_fb15k-237.json'
	'configs/reasoners/qto_nell.json',
	'configs/reasoners/fit_nell.json'
] 

for config_path in config_paths:
	print(config_path, '\n')
	test_reasoner(config_path)
	print('==============================================')