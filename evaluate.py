import torch
import numpy as np
import datetime
import json
from tqdm import tqdm

from argparse import ArgumentParser

from torch.utils.data import DataLoader

from src.utils.metric_utils import evaluate_mrr, evaluate_class_metrics
from src.csp.cq_data import CQ_Data
from src.model.model import ANYCQ
from src.data.dataset import CQA_Dataset
from src.utils.config_utils import dataset_from_config


if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument("--model_dir", type=str, help="Model directory")
	parser.add_argument("--model_name", type=str, help="Model name")
	parser.add_argument("--config_dir", type=str, help="Config directory")
	parser.add_argument("--timeout", type=int, default=1200, help="Timeout in seconds")
	parser.add_argument("--exp_name", type=str, default='')
	args = parser.parse_args()
	dict_args = vars(args)

	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	with open(args.config_dir, 'r') as f:
		model_config = json.load(f)
	dataset_config = model_config['val_data']

	model = ANYCQ(None, model_config)
	model = model.load_model(args.model_dir, name = args.model_name)
	model.to(device)
	model.eval()

	exp_name = args.exp_name+"".join(str(datetime.datetime.now()).split(' '))

	with open('logs/'+exp_name+'.out', 'w') as f:
		f.write('Starting evaluation:\n\n')


	for i, dataset_name in enumerate(dataset_config["data_files"]):

		dataset_config['data_file'] = dataset_name

		dataset = dataset_from_config(dataset_config, device)

		val_loader = DataLoader(
			dataset,
			batch_size=model_config['val_batch_size'][i],
			num_workers=0,
			collate_fn=CQ_Data.collate,
			shuffle = True
		)

		cq_type = dataset.cq_types[0]

		with open('logs/'+exp_name+'.out', 'a') as f:
			f.write(cq_type+'\n')

		total_easy_prec = 0
		total_hard_prec = 0
		total_easy_rec = 0
		total_hard_rec = 0
		total_easy_f1 = 0
		total_hard_f1 = 0
		total_em = 0
		total_rec_rank_easy = 0
		total_rec_rank_hard = 0

		for in_data in tqdm(val_loader, total=len(val_loader)):

			with torch.no_grad():
				out_data = model(in_data,
								model_config['T_val'],
								return_all_scores=True,
								return_all_assignments = True
								)

			scores = out_data.all_cq_scores.cpu()
			assgns = out_data.all_assignments.cpu()

			pred_mask = torch.zeros((scores.shape[0], dataset.n_entities))
			easy_mask = torch.zeros((scores.shape[0], dataset.n_entities))
			hard_mask = torch.zeros((scores.shape[0], dataset.n_entities))

			predictions = [torch.unique(assgns[in_data.pred_mask.cpu()][i][scores[i]>0]) for i in range(scores.shape[0])]
			correct_ans_easy = in_data.corr_ans_easy[0]
			correct_ans_hard = in_data.corr_ans_hard[0]
			correct_ans_easy = [torch.Tensor(answr).flatten().long() for answr in correct_ans_easy]
			correct_ans_hard = [torch.Tensor(answr).flatten().long() for answr in correct_ans_hard]

			for i in range(scores.shape[0]):
				pred_mask[i][predictions[i]] = 1
				easy_mask[i][correct_ans_easy[i]] = 1
				easy_mask[i][correct_ans_hard[i]] = 1
				hard_mask[i][correct_ans_hard[i]] = 1

			sorted_ids = [torch.unique_consecutive(assgns[in_data.pred_mask.cpu()][i][torch.argsort(scores[i], descending=True)]) for i in range(scores.shape[0])]

			batch_easy_rec, batch_easy_pre, batch_hard_rec, batch_hard_pre, batch_em = evaluate_class_metrics(pred_mask, easy_mask, hard_mask, at_T=model_config['T_val'])
			rec_ranks_easy,	rec_ranks_hard = evaluate_mrr(sorted_ids, correct_ans_easy, correct_ans_hard)

			batch_easy_f1 = 2 / (1/batch_easy_pre + 1/batch_easy_rec)
			batch_hard_f1 = 2 / (1/batch_hard_pre + 1/batch_hard_rec)
			batch_easy_f1[batch_easy_pre==0] = 0
			batch_easy_f1[batch_easy_rec==0] = 0
			batch_hard_f1[batch_hard_pre==0] = 0
			batch_hard_f1[batch_hard_rec==0] = 0

			total_rec_rank_easy += rec_ranks_easy.sum()
			total_rec_rank_hard += rec_ranks_hard.sum()
			total_easy_prec += batch_easy_pre.sum()
			total_hard_prec += batch_hard_pre.sum()
			total_easy_rec 	+= batch_easy_rec.sum()
			total_hard_rec 	+= batch_hard_rec.sum()
			total_easy_f1 	+= batch_easy_f1.sum()
			total_hard_f1 	+= batch_hard_f1.sum()
			total_em += batch_em.sum()
		
		with open('logs/'+exp_name+'.out', 'a') as f:
			f.write('Easy MRR: '+ str((total_rec_rank_easy).item()/len(dataset))+'\n')
			f.write('Hard MRR: '+ str((total_rec_rank_hard).item()/len(dataset))+'\n')
			f.write('Easy Precision: '+ str((total_easy_prec).item()/len(dataset))+'\n')
			f.write('Easy Recall:    '+ str((total_easy_rec).item()/len(dataset))+'\n')
			f.write('Easy F1-score:  '+ str((total_easy_f1).item()/len(dataset))+'\n')
			f.write('Hard Precision: '+ str((total_hard_prec).item()/len(dataset))+'\n')
			f.write('Hard Recall:    '+ str((total_hard_rec).item()/len(dataset))+'\n')
			f.write('Hard F1-score:  '+ str((total_hard_f1).item()/len(dataset))+'\n')
			f.write('Exact Match: '+ str((total_em).item()/len(dataset))+'\n\n')