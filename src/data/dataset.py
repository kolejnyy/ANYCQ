import json
import ijson
import os
import numpy as np
import pandas as pd
from copy import deepcopy

import torch
from torch.utils.data import Dataset

from src.csp.cq_data import CQ_Data
from src.utils.data_utils import load_encodings, baseline_nx, ent_type, augment



class CQA_Dataset(Dataset):

    def __init__(self, device):
        # A dictionary containing data 
        self.data = {}
        # Types of data available in the dictionary
        self.cq_types = []
        # Numbers of queries of given type 
        self.cq_nums = []
        # and the corresponding offset in the data
        self.cq_off = []

        # Entity and relation encodings
        self.encode_e = {}
        self.encode_r = {}

        # Number of different entities in the dataset
        self.n_entities = 0
        self.n_relations = 0

        # Predictor associated with the dataset
        self.predictor = None

        self.device = device

    def load(self, path, predictor, data_name='train_qaa.json', PE_path = None):

        self.predictor = predictor.to(self.device)

        # Load the data
        with open(os.path.join(path, data_name), 'rb') as f:
            for qitem in ijson.items(f, ''):
                self.cq_types = list(qitem.keys())
                self.data = qitem

        # Load relation and entity encodings
        self.encode_r, self.encode_e = load_encodings(path)
        self.n_entities = len(self.encode_e.keys())
        self.n_relations = len(self.encode_r.keys())

        # Evaluate the number of queries of given types and corresponding offsets
        self.cq_nums = np.array([len(self.data[qtype]) for qtype in self.cq_types])
        self.cq_off = np.concatenate(([0],np.cumsum(self.cq_nums)[:-1]))
        self.length = self.cq_nums[-1] + self.cq_off[-1]

        # Define the baseline networkx graphs for each query type
        self.sample_nx = {cq_type: baseline_nx(cq_type) for cq_type in self.cq_types}

        graph = pd.read_csv(os.path.join(path, 'train_kg.tsv'), sep='\t')

        self.poss_heads_map = torch.zeros((self.n_relations, self.n_entities))
        self.poss_tails_map = torch.zeros((self.n_relations, self.n_entities))

        if PE_path is None:
            possible_heads = [pd.unique(graph[graph['0.1']==rel_id]['0']) for rel_id in range(self.n_relations)]
            possible_tails = [pd.unique(graph[graph['0.1']==rel_id]['1']) for rel_id in range(self.n_relations)]
            for rel_id in range(self.n_relations):
                self.poss_heads_map[rel_id][torch.from_numpy(possible_heads[rel_id])] = 1
                self.poss_tails_map[rel_id][torch.from_numpy(possible_tails[rel_id])] = 1
        
        else:
            self.poss_heads_map = torch.load(PE_path + 'PE_head.pt', map_location=torch.device('cpu')).to('cpu').to_dense()
            self.poss_tails_map = torch.load(PE_path + 'PE_tail.pt', map_location=torch.device('cpu')).to('cpu').to_dense()


    def __len__(self):
        return self.length

    def __getitem__(self, item):
        # Find the type and index of the (item)'th query
        qc_type = np.argmax(self.cq_off>item)-1
        index = item - self.cq_off[qc_type]
        qc_type = self.cq_types[qc_type]

        # Create a specific version of nx graph for the instance
        spec_nx  = augment(self.sample_nx[qc_type], self.data[qc_type][index][0])
        
        # Convert the instance to CQ_Data object
        free_rep = list(self.data[qc_type][index][1].keys())[0]
        corr_ans_easy = self.data[qc_type][index][1][free_rep]
        corr_ans_hard = []
        if len(self.data[qc_type][index]) > 2 and free_rep in self.data[qc_type][index][2]:
            corr_ans_hard = self.data[qc_type][index][2][free_rep]
        cq_data = self.nx_to_cq(spec_nx, corr_ans_easy, corr_ans_hard)

        return cq_data


    def get_grounded(self, item, const_val):
        # Find the type and index of the (item)'th query
        qc_type = np.argmax(self.cq_off>item)-1
        index = item - self.cq_off[qc_type]
        qc_type = self.cq_types[qc_type]

        new_qc_type = qc_type.replace('f1', 's0')
        new_input_dict = self.data[qc_type][index][0]
        new_input_dict['s0'] = const_val

        # Create a specific version of nx graph for the instance
        spec_nx  = augment(baseline_nx(new_qc_type), new_input_dict)
        
        # Convert the instance to CQ_Data object
        corr_ans_easy = []
        corr_ans_hard = []
        cq_data = self.nx_to_cq(spec_nx, corr_ans_easy, corr_ans_hard)

        return cq_data


    def nx_to_cq(self, aug_graph, corr_ans_easy, corr_ans_hard):

        num_var = aug_graph.number_of_nodes()
        num_cst = aug_graph.number_of_edges()
        
        domain_size = torch.Tensor([self.n_entities if aug_graph.nodes[u]['type']!='const' else 1 for u in range(num_var)]).long()
        const_val	= torch.Tensor([aug_graph.nodes[u]['idx'] if aug_graph.nodes[u]['type']=='const' else 0 for u in range(num_var)]).long()
        pred_mask   = torch.Tensor([True if aug_graph.nodes[u]['type']=='pred' else False for u in range(num_var)]).bool()
        # Create the CQ_Data structure
        cq_data = CQ_Data(num_var, domain_size, const_val, pred_mask, self.predictor, corr_ans_easy, corr_ans_hard, device=self.device)

        # Negation mask
        cst_type = torch.Tensor([1 if aug_graph.edges[u,v]['neg'] else 0 for u,v in aug_graph.edges])
        # Disjunction mask
        dis_part = torch.Tensor([1 if 'dis_part' in aug_graph.edges[u,v] else 0 for u,v in aug_graph.edges]).long()
        # Variable offset
        var_off = cq_data.var_off

        # Create constraint edges
        cst_edges_zero = torch.repeat_interleave(
            torch.arange(num_cst),
            torch.Tensor([domain_size[u]+domain_size[v] for u,v in aug_graph.edges]).long()
        ).long()
        cst_edges_one = [
            list(range(var_off[u], var_off[u]+domain_size[u]))+list(range(var_off[v], var_off[v]+domain_size[v]))
            for u,v in aug_graph.edges
        ]
        cst_edges_one = torch.Tensor([item for arr in cst_edges_one for item in arr]).long()
        cst_edges = torch.stack([cst_edges_zero, cst_edges_one]).long()

        # Create head masks
        head_mask = torch.repeat_interleave(
            torch.Tensor([True, False]*num_cst).bool(),
            torch.Tensor([[domain_size[u],domain_size[v]] for u,v in aug_graph.edges]).flatten().long()
        )

        # Constraint, head and tail types
        head_ids = torch.Tensor([u for u,v in aug_graph.edges]).long()
        rel_ids = torch.Tensor([aug_graph.edges[u,v]['idx'] for u,v in aug_graph.edges]).long()
        tail_ids = torch.Tensor([v for u,v in aug_graph.edges]).long()

        val_edges = cq_data.val_ids[cst_edges_one].long().cpu()

        PE = self.poss_heads_map[rel_ids[cst_edges_zero], val_edges]*head_mask + self.poss_tails_map[rel_ids[cst_edges_zero], val_edges]*(~head_mask)
        PE[cst_type[cst_edges[0]]==1] = 1
        PE = PE.long()  

        cq_data.add_constraint_data(
            cst_type,
            dis_part,
            cst_edges,
            head_mask,
            head_ids,
            rel_ids,
            tail_ids,
            PE
        )

        return cq_data
