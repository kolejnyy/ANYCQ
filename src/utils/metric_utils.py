import torch
import json
import os

def evaluate_mrr(sorted_ids, correct_ans_easy, correct_ans_hard):

	rec_ranks_easy = []
	rec_ranks_hard = []

	for ex_id in range(len(sorted_ids)):
		rank_easy = 1
		rank_hard = 1
		
		found_easy = False
		found_hard = False

		considered = []

		for pred_id in sorted_ids[ex_id]:
			if pred_id in considered:
				continue
			considered.append(pred_id)
			if pred_id in correct_ans_easy[ex_id]:
				if not found_easy:
					rec_ranks_easy.append(1/rank_easy)
				found_easy = True
				continue
			if pred_id in correct_ans_hard[ex_id]:
				rec_ranks_hard.append(1/rank_hard)
				found_hard=True
				if not found_easy:
					rec_ranks_easy.append(1/rank_easy)
				found_easy=True
				break

			rank_easy += 1
			rank_hard += 1

		if not found_easy:
			rec_ranks_easy.append(0)
		if not found_hard:
			rec_ranks_hard.append(0)
	
	rec_ranks_easy = torch.Tensor(rec_ranks_easy)
	rec_ranks_hard = torch.Tensor(rec_ranks_hard)


	return rec_ranks_easy, rec_ranks_hard




def validate_predictor_mrr(model, dataset_name):

	with open(os.path.join('data', dataset_name, 'valid_type0000_real_EFO1_qaa.json'), 'rb') as f:
		valid_data = json.load(f)['r1(s1,f1)']

	# Initialize the counters
	mean_mrr_hd = 0
	total_num = 0

	# Iterate over all queries
	for csts, ans_1, ans_2 in valid_data:
		
		corr_ans_easy = torch.Tensor(ans_1['f1']).flatten().long()
		corr_ans_hard = torch.Tensor(ans_2['f1']).flatten().long()

		# Evaluate the scores
		scores = model.tail_scores([csts['s1']], [csts['r1']])
		_, indices = torch.sort(scores, dim=1, descending=True)
		indices = indices.cpu().flatten()

		ez_ans_map = torch.zeros(100000)
		ez_ans_map[corr_ans_easy] = 1
		ez_ans_pos = ez_ans_map[indices]
		hd_ans_map = torch.zeros(100000)
		hd_ans_map[corr_ans_hard] = 1
		hd_ans_pos = hd_ans_map[indices]

		pos_scores = torch.arange(len(indices)) - torch.cumsum(ez_ans_pos, dim=0) + 1
		pos_scores = pos_scores[hd_ans_pos==1]
		pos_scores = 1 / pos_scores

		mean_mrr_hd += pos_scores.max()

		total_num += 1

	mean_mrr_hd /= total_num
	
	return mean_mrr_hd



def validate_predictor_accuracy(model, dataset_name, print_res = False):

	val_data_easy = torch.load(os.path.join('data', dataset_name, 'predictor_val_easy.pt'))
	val_data_hard = torch.load(os.path.join('data', dataset_name, 'predictor_val_hard.pt'))

	with torch.no_grad():
		pred_easy = model(val_data_easy[0], val_data_easy[1], val_data_easy[2])
		pred_hard = model(val_data_hard[0], val_data_hard[1], val_data_hard[2])

	acc_easy = ((pred_easy>0.5).sum()/len(pred_easy)).item()*100
	acc_hard = ((pred_hard>0.5).sum()/len(pred_hard)).item()*100
	
	with torch.no_grad():
		pred_neg_tail = model(val_data_easy[0], val_data_easy[1], torch.randint(val_data_easy[2].max()+1, val_data_easy[2].shape))
		pred_neg_head = model(torch.randint(val_data_easy[0].max()+1, val_data_easy[0].shape), val_data_easy[1], val_data_easy[2])

	acc_neg = (((pred_neg_tail<0.5).sum()/len(pred_neg_tail)).item()+((pred_neg_head<0.5).sum()/len(pred_neg_head)).item())*50

	if print_res:
		print(f"Accuracy on positive easy examples:  {acc_easy:.3f} %")
		print(f"Accuracy on positive hard examples:  {acc_hard:.3f} %")
		print(f"Accuracy on negative examples:       {acc_neg:.3f} %")

	return acc_easy, acc_hard, acc_neg




def evaluate_class_metrics(_pred_mask, _easy_mask, _hard_mask, at_T = 1000000):

	_at_T = torch.zeros(_easy_mask.shape[0]) + at_T

	_easy_rec = (_pred_mask*_easy_mask).sum(dim=1) / torch.min((_easy_mask).sum(dim=1), _at_T)
	_hard_rec = (_pred_mask*_hard_mask).sum(dim=1) / torch.min((_hard_mask).sum(dim=1), _at_T)
	_easy_pre = (_pred_mask*_easy_mask).sum(dim=1) / (_pred_mask).sum(dim=1)
	_hard_pre = (_pred_mask*_hard_mask).sum(dim=1) / ((_pred_mask).sum(dim=1)-(_pred_mask*_easy_mask).sum(dim=1))

	_hard_rec[_hard_mask.sum(dim=1) == 0] = 0
	_easy_pre[_pred_mask.sum(dim=1) == 0] = 0
	_hard_pre[_pred_mask.sum(dim=1) == 0] = 0
	_hard_pre[_pred_mask.sum(dim=1) == (_pred_mask*_easy_mask).sum(dim=1)] = 0
	
	_em = (_pred_mask == _easy_mask).min(dim=1)[0].long()

	return _easy_rec, _easy_pre, _hard_rec, _hard_pre, _em