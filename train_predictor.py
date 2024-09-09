import json
import ijson

import os
import re
from copy import deepcopy

import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset

import datetime
from time import time
from tqdm import tqdm

from argparse import ArgumentParser

from src.predictor.complex import ComplEx
from src.utils.metric_utils import validate_predictor_mrr, validate_predictor_accuracy
from src.utils.loss_utils import evaluate_n3_loss, get_distributions


def load_encodings(path):
	with open(os.path.join(path, 'kgindex.json')) as f:
		map_dict = json.load(f)

	encode_r = {}
	encode_e = {}


	for key in map_dict['r'].keys():
		encode_r[map_dict['r'][key]] = key
	for key in map_dict['e'].keys():
		encode_e[map_dict['e'][key]] = key

	return encode_r, encode_e



def predictor_training(model_type, dataset_name, epochs, batch_size, learning_rate, embedding_dim, lmbda, w_r, val_interval=10, save_interval=1):

	encode_r, encode_e = load_encodings('data/'+dataset_name)
	n_entities = len(encode_e.keys())
	logs_path = os.path.join('logs', model_type+'_'+dataset_name+'_'+"_".join(str(datetime.datetime.now()).split(' ')))

	save_dir = os.path.join('models', dataset_name, 'predictor')
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)

	model = None
	if model_type == 'ComplEx':
		model = ComplEx(len(encode_r.keys()), len(encode_e.keys()), embedding_dim, 'cuda')

	optimizer = torch.optim.Adagrad(model.parameters(), lr = learning_rate)

	with open(logs_path, 'w') as f:
		f.write('Starting training\n')
		f.write("Model type: " + model_type + '\n')
		f.write("Dataset: " + dataset_name + "\n")
		f.write("Epochs: " + str(epochs) + '\n')
		f.write("LR: " + str(learning_rate) + '\n')

	data_length = dataset.shape[1]

	for ep_id in tqdm(range(epochs)):

		total_head_loss = 0
		total_tail_loss = 0
		total_rel_loss = 0
		total_reg_loss = 0
		ep_num = 0

		for i in range(0, data_length, batch_size):

			b_size = min(batch_size, data_length-i)
			reg_const = lmbda/3

			heads 	= dataset[0][i:i+b_size]
			rels	= dataset[1][i:i+b_size]
			tails 	= dataset[2][i:i+b_size]			

			optimizer.zero_grad()

			l_h, l_r, l_t, l_reg = evaluate_n3_loss(model, heads, rels, tails, dist_h.cuda(), dist_r.cuda(), dist_t.cuda())

			# Evaluate the cumulative loss
			loss = l_h + l_t + w_r * l_r + reg_const * l_reg

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			
			total_head_loss += l_h.item()
			total_tail_loss += l_t.item()
			total_rel_loss += l_r.item()
			total_reg_loss += reg_const*l_reg.item()
			ep_num += b_size/1000
		
		best_mrr = 0

		with open(logs_path, 'a') as f:
			f.write("Epoch "+str(ep_id)+":   head loss = "+str(total_head_loss/ep_num)+'      rel loss = '+str(total_rel_loss/ep_num)
					+'      tail loss = '+str(total_tail_loss/ep_num)+'      reg loss = '+str(total_reg_loss/ep_num)+'\n')

			if ep_id%val_interval == val_interval-1:
				acc_easy, acc_hard, acc_neg = validate_predictor_accuracy(model, dataset_name)
				f.write("\n"+f"Accuracy on positive easy examples:  {acc_easy:.3f} %\n"
							+f"Accuracy on positive hard examples:  {acc_hard:.3f} %\n"
							+f"Accuracy on negative examples:       {acc_neg:.3f} %\n")

				mean_mrr = validate_predictor_mrr(model, dataset_name)
				f.write(	f"MRR:      {100*mean_mrr:.3f} %\n"
							+"\n")

				if mean_mrr > best_mrr:
					torch.save(model.state_dict(), os.path.join('models', dataset_name, 'predictor', model_type+'_best.pth'))
					best_mrr = mean_mrr

		if ep_id % save_interval == save_interval-1:
			torch.save(model.state_dict(), os.path.join('models', dataset_name, 'predictor', model_type+'_epoch_'+str(ep_id+1)+'.pth'))

	torch.save(model.state_dict(), os.path.join('models', dataset_name, 'predictor', model_type+'_last.pth'))
	


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("--config", type=str, help="path the config file")
	args = parser.parse_args()

	with open(args.config) as f:
		config = json.load(f)

	epochs 		= config['epochs']
	batch_size	= config['batch_size']
	embd_dim	= config['embedding_dim']
	lr			= config['lr']
	lmbda		= config['reg_lambda']
	w_r			= config['rel_lambda']
	model_type	= config['model']
	data_name	= config['dataset_name']
	val_interv	= config['val_interval']
	save_interv	= config['save_interval']

	dataset = torch.load(os.path.join('data', data_name, 'predictor_train.pt')).long()
	dist_h, dist_r, dist_t = get_distributions(dataset)
	dataset = dataset.cuda()
	predictor_training(model_type, data_name, epochs, batch_size, lr, embd_dim, lmbda, w_r, val_interval=val_interv, save_interval = save_interv)
