import os
import json
import torch
from torch import nn
from tqdm import tqdm
from argparse import ArgumentParser

from src.utils.config_utils import predictor_from_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'


config_paths = [
	# 'configs/reasoners/fit_fb15k-237.json',
	# 'configs/reasoners/qto_fb15k-237.json',
	'configs/reasoners/qto_nell.json',
	# 'configs/reasoners/fit_nell.json'
] 


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("--temp", type=float, help="Temperature")
	args = parser.parse_args()
	dict_args = vars(args)

	for config_path in config_paths:

		with open(config_path, 'r') as f:
			pred_config = json.load(f)

		batch_size		= 1000
		dataset_name 	= pred_config['dataset']
		scaling_rule 	= pred_config['predictor']['scaling_rule']
		n_entities 		= pred_config['n_entities']
		n_relations 	= pred_config['n_relations']
		pred_config['predictor']['temperature'] = dict_args['temp']
		predictor = predictor_from_config(pred_config['predictor'], device)

		path = 'models/'+dataset_name+'/reasoner_test_'+str(dict_args['temp'])+'/'+scaling_rule
		if not os.path.exists(path):
			os.makedirs(path)

		for rel_id in tqdm(range(n_relations)):

			neural_adj = torch.zeros((n_entities, n_entities))

			for i in (range(0, n_entities, batch_size)):
				j = min(n_entities-i, batch_size)
				preds = predictor.tail_scores(torch.arange(i,i+j), torch.zeros(j)+rel_id)
				preds[preds < 0.01] = 0
				neural_adj[i:i+j] = preds.cpu()

			torch.save(neural_adj.to_sparse(), path+'/'+'P_'+str(rel_id)+'.pt')