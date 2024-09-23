import torch
import json
from time import time

import pandas as pd
from tqdm import tqdm
from src.utils.config_utils import predictor_from_config



with open('configs/model/model_NELL_qar.json', 'r') as f:
	config = json.load(f)

predictor_config = config['val_data']['predictor']['predictor']
predictor = predictor_from_config(predictor_config, 'cuda')

logZ = torch.zeros((400, 63361), device='cuda')
logD = torch.zeros((400, 63361), device='cuda')


train_kg = pd.read_csv('data/NELL-QAR/train_kg.tsv', sep = '\t', header=None)
valid_kg = pd.read_csv('data/NELL-QAR/valid_kg.tsv', sep = '\t', header=None)

E = torch.zeros((400, 63361), device='cuda')

for x in tqdm(train_kg.iloc):
	E[x[1], x[0]] += 1

for i in tqdm(range(400)):
	for j in range(0, 63361, 10000):
		k = min(63361, j+10000)

		with torch.no_grad():
			x = (predictor.tail_scores(torch.arange(j, k), torch.zeros(k-j)+i))
		logZ[i, j:k] = torch.logsumexp(x, dim=1)
		masked_x = torch.zeros_like(x, device='cuda') - 10000
		for y in (train_kg[train_kg[1]==i].iloc):
			a = y[0]; b = y[2]
			if a >= j and a < k:
				masked_x[a-j,b] = x[a-j,b]
		logD[i, j:k] = torch.logsumexp(masked_x, dim=1)

E[E==0] = 1e-12
logE = E.log()
logDelta = logE - logD
logDelta[E<1] = -logZ[E<1]

torch.save(logDelta, 'models/NELL-EFO1/complex/logDelta_t1.pt')