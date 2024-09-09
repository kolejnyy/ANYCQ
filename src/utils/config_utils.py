import os
import json
import torch

from src.data.dataset import CQA_Dataset
from src.predictor.complex import ComplEx
from src.predictor.perfect import PerfeCT
from src.predictor.cqpred import CQPred, SymCQPred, BinCQPred, SignCQPred

def exists(path):
    path = os.path.join(path, 'config.json')
    return os.path.exists(path)


def write_config(config, path):
    path = os.path.join(path, 'config.json')
    with open(path, 'w') as f:
        json.dump(config, f, indent=4)


def read_config(path):
    with open(path, 'r') as f:
        conf_dict = json.load(f)
    return conf_dict


def predictor_from_config(pred_config, device):

    predictor = None

    if pred_config['type'] == 'ComplEx':
        predictor = ComplEx(
            n_relations = pred_config['n_relations'],
            n_entities = pred_config['n_entities'],
            embedding_dim = pred_config['embedding_dim'],
            device = device,
            tau = pred_config['tau']
        )

        if 'load_path' in pred_config:
            predictor.load_state_dict(torch.load(pred_config['load_path'], map_location=torch.device(device)))

    elif pred_config['type'] == 'PerfeCT':
        predictor = PerfeCT(
            data_path = pred_config['graph_path'],
            device = device
        )

    elif pred_config['type'] == 'CQPred':
        predictor = CQPred(
            pred_config['perfect'],
            pred_config['predictor'],
            device,
            pred_config['scaling_rule'],
            pred_config['eps'],
            pred_config['temperature']
        )
    
    elif pred_config['type'] == 'SymCQPred':
        predictor = SymCQPred(
            pred_config['perfect'],
            pred_config['predictor'],
            pred_config['logDelta'],
            device,
            pred_config['eps'],
            pred_config['temperature'],
            pred_config['threshold']
        )

    elif pred_config['type'] == 'BinCQPred':
        predictor = BinCQPred(
            pred_config['perfect'],
            pred_config['predictor'],
            device,
            pred_config['scaling_rule'],
            pred_config['eps'],
            pred_config['temperature'],
            pred_config['threshold']
        )
    
    elif pred_config['type'] == 'SignCQPred':
        predictor = SignCQPred(
            pred_config['perfect'],
            pred_config['predictor'],
            device,
            pred_config['scaling_rule'],
            pred_config['eps'],
            pred_config['temperature'],
            pred_config['threshold']
        )
    
    else:
        raise NotImplementedError

    return predictor


def dataset_from_config(data_config, device):
    
    # Initialize the dataset
    dataset = CQA_Dataset(device) 

    # Create and load the predictor
    predictor = predictor_from_config(data_config['predictor'], device)

    # Load the dataset
    dataset.load(
        data_config['folder_path'],
        predictor,
        data_name = data_config['data_file'],
        PE_path = data_config['predictor']['PE_path'] if 'PE_path' in data_config['predictor'] else None 
    )    

    return dataset    
