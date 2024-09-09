import numpy as np
import torch
from time import time
from torch_geometric.utils import degree
from torch_scatter import scatter_softmax, scatter_sum, scatter_max, scatter_min

from src.csp.constraints import CQ_Constraint_Data


# Class for representing CQ instances for torch
class CQ_Data:

    def __init__(self, num_var, domain_size, const_val, pred_mask, predictor, corr_ans_easy, corr_ans_hard, domain=None, batch=None, path=None, device='cpu'):

        # The path to the file from which the instance was parsed (not used anywhere yet)
        self.path = path
        # The number of variables
        self.num_var = num_var
        # The number of constraints
        self.num_cst = 0
        # Device
        self.device = device

        # Change the domain_size from Int to Tensor, if necesary
        if isinstance(domain_size, torch.Tensor):
            self.domain_size = domain_size
        else:
            self.domain_size = domain_size * torch.ones((num_var,), dtype=torch.int64)
        self.domain_size = self.domain_size.to(device)

        # Number of values in all domains
        self.num_val = int(self.domain_size.sum().cpu().numpy())
        # The maximum size of a domain
        self.max_dom = int(self.domain_size.max().cpu().numpy())

        # Constant value entities
        self.is_const = self.domain_size==1
        self.const_val = const_val.to(device)
        self.const_val[self.is_const==False] = 0

        # Mask for free variables
        self.pred_mask = pred_mask

        # Correct answers
        self.corr_ans_easy = [corr_ans_easy]
        self.corr_ans_hard = [corr_ans_hard]

        # Single-link predictor
        self.predictor = predictor

        # The mapping of value index to the index of the corresponding variable
        self.var_idx = torch.repeat_interleave(torch.arange(0, num_var, device=self.device, dtype=torch.int64), self.domain_size)
        # The offset for each variable (the first index of a value corresponding to the variable)
        self.var_off = torch.cat([torch.tensor([0], device=self.device), torch.cumsum(self.domain_size, dim=0)[:-1]], dim=0)
        # Indices of values within the domain of the corresponding variable 
        self.dom_idx = torch.arange(self.num_val, device = self.device) - self.var_off[self.var_idx]
        # Same as above, but considering constant values
        self.val_ids = (self.dom_idx + self.const_val[self.var_idx]).long()

        # Setting up the domain, to either the predefined values or self.dom_idx
        if domain is not None:
            self.domain = domain
        else:
            self.domain = torch.arange(self.num_val, device=self.device) - self.var_off[self.var_idx]

        # Stores the inverse of the domain size of the variable corresponding to a value with given idx
        self.var_reg = 1.0 / (self.domain_size + 1.0e-8).view(-1, 1)
        self.cst_reg = None
        self.val_reg = None

        # Batch processing
        self.batch = torch.zeros((num_var,), device = self.device, dtype=torch.int64) if batch is None else batch
        self.batch_size = int(self.batch.max().cpu().numpy()) + 1
        self.batch_num_cst = torch.zeros((self.batch_size,), device = self.device, dtype=torch.float32)

        # Number of values within each batch
        self.batch_num_val = scatter_sum(self.domain_size, self.batch, dim=0)
        # Offsets within each batch
        self.batch_val_off = torch.zeros((self.batch_size,), device = self.device, dtype=torch.long)
        self.batch_val_off[1:] = torch.cumsum(self.batch_num_val[:-1], dim=0)
        # self.dom_idx but for batch
        self.batch_val_idx = torch.arange(self.num_val, device = self.device)
        self.batch_val_idx -= self.batch_val_off[self.batch[self.var_idx]]
        # Maximum number of values in a batch
        self.max_num_val = self.batch_num_val.max().cpu().numpy()

        # Counters
        self.num_edges = 0
        self.constraints = {}
        self.cst_batch = None
        self.cst_edges = None
        self.LE = None
        self.PE = None

        self.initialized = False

    def to(self, device):
        self.device = device
        self.domain_size = self.domain_size.to(device)
        self.is_const = self.is_const.to(device)
        self.const_val = self.const_val.to(device)
        self.predictor = self.predictor.to(device)
        self.var_idx = self.var_idx.to(device)
        self.var_off = self.var_off.to(device)
        self.dom_idx = self.dom_idx.to(device)
        self.val_ids = self.val_ids.to(device)
        self.domain = self.domain.to(device)
        self.pred_mask = self.pred_mask.to(device)
        self.var_reg = self.var_reg.to(device)

        self.batch = self.batch.to(device)
        self.batch_num_val = self.batch_num_val.to(device)
        self.batch_val_off = self.batch_val_off.to(device)
        self.batch_val_idx = self.batch_val_idx.to(device)
        self.batch_num_cst = self.batch_num_cst.to(device)

        self.device = device

        for cst_data in self.constraints.values():
            cst_data.to(device)

    # Given a list of CQ_Data instances, create a combined batch in the form of a single CQ_Data
    @staticmethod
    def collate(batch):
        num_var = [d.num_var for d in batch]
        var_off = np.concatenate([[0], np.cumsum(num_var)[:-1]], axis=0)
        num_val = [d.num_val for d in batch]
        val_off = np.concatenate([[0], np.cumsum(num_val)[:-1]], axis=0)

        predictor = batch[0].predictor

        num_var = sum(num_var)

        domain_size = torch.cat([d.domain_size for d in batch])
        domain = torch.cat([d.domain for d in batch])

        corr_ans_easy = []
        corr_ans_hard = []
        for data in batch:
            corr_ans_easy = corr_ans_easy + data.corr_ans_easy
            corr_ans_hard = corr_ans_hard + data.corr_ans_hard

        const_val = torch.cat([d.const_val for d in batch])
        pred_mask = torch.cat([d.pred_mask for d in batch])

        # Now, that's a bit weird way to handle batch_idx?
        batch_idx = torch.cat([d.batch + i for i, d in enumerate(batch)])
        batch_data = CQ_Data(num_var, domain_size, const_val, pred_mask, predictor, corr_ans_easy, corr_ans_hard, domain=domain, batch=batch_idx, device=batch[0].device)

        cst_batch_dict = {}
        for i, data in enumerate(batch):
            for key, cst_data in data.constraints.items():
                batch_item = (cst_data, var_off[i], val_off[i], i)
                if key in cst_batch_dict:
                    cst_batch_dict[key].append(batch_item)
                else:
                    cst_batch_dict[key] = [batch_item]

        for key, batch_list in cst_batch_dict.items():
            const_data = batch_list[0][0].collate(batch_list, batch_data)
            batch_data.add_constraint_data_(const_data, key)

        batch_data.to(batch[0].device)

        return batch_data

    # Update the constraint information after adding constraints
    def init_adj(self):
        cst_edges, cst_batch = [], []
        PEs = []
        cst_off = 0
        for cst_data in self.constraints.values():
            cur_edges = cst_data.cst_edges.clone()
            cur_edges[0] += cst_off
            cst_edges.append(cur_edges)
            PEs.append(cst_data.PE)

            cst_batch.append(cst_data.batch)
            self.num_edges += cst_data.num_edges
            cst_off += cst_data.num_cst

        self.PE = torch.cat(PEs, dim=0).long()
        self.cst_edges = torch.cat(cst_edges, dim=1)
        self.cst_batch = torch.cat(cst_batch, dim=0)
        self.batch_num_cst = degree(self.cst_batch, num_nodes=self.batch_size, dtype=torch.int64)

    def update_LE(self):
        self.LE = torch.cat([cst_data.LE for cst_data in self.constraints.values()], dim=0).flatten().long()

    def add_constraint_data_(self, cst_data, name):
        self.num_cst += cst_data.num_cst
        if name not in self.constraints:
            self.constraints[name] = cst_data
        else:
            cst_old = self.constraints[name]
            cst_data = cst_data.collate([(cst_old, 0, 0, 0), (cst_data, 0, 0, 0)], self)
            self.constraints[name] = cst_data

    def add_constraint_data(self, cst_type, dis_part, cst_edges, head_mask, head_ids, rel_ids, tail_ids, PE, batch = None):

        cst_data = CQ_Constraint_Data(
            cq_data=self,
            cst_type=cst_type,
            dis_part = dis_part,
            cst_edges = cst_edges,
            head_mask = head_mask,
            head_ids = head_ids,
            rel_ids = rel_ids,
            tail_ids = tail_ids,
            batch = batch,
            PE = PE
        )
        cst_data.to(self.device)

        self.add_constraint_data_(cst_data, 'ext')

    def value_softmax(self, value_logits):
        with torch.cuda.amp.autocast(enabled=False):
            value_logits = value_logits.view(self.num_val, -1).float()
            value_max = scatter_max(value_logits, self.var_idx, dim=0)[0]
            value_logits = value_logits - value_max[self.var_idx]
            value_logits = torch.clip(value_logits, -100, 0)
            value_prob = scatter_softmax(value_logits, self.var_idx, dim=0)
        return value_prob

    def value_softmax_local(self, value_logits, cur_assignment):
        with torch.cuda.amp.autocast(enabled=False):
            value_logits = value_logits.float().view(self.num_val, -1)
            value_logits -= 10000.0 * cur_assignment.view(self.num_val, -1)
            value_prob = scatter_softmax(value_logits.float(), self.batch[self.var_idx], dim=0)
        return value_prob

    def round_to_one_hot(self, value_prob):
        value_prob = value_prob.view(self.num_val, -1)
        max_idx = scatter_max(value_prob, self.var_idx, dim=0)[1]
        step_idx = torch.arange(value_prob.shape[1], device=value_prob.device).view(1, -1)
        one_hot = torch.zeros_like(value_prob)
        one_hot[max_idx, step_idx] = 1.0
        return one_hot

    def hard_assign_max(self, value_prob):
        value_prob = value_prob.view(self.num_val, -1)
        value_idx = scatter_max(value_prob, self.var_idx, dim=0)[1]
        value_idx -= self.var_off.view(-1, 1)
        return value_idx

    def hard_assign_sample(self, logits):
        value_prob = self.value_softmax(logits).view(self.num_val, 1)
        with torch.no_grad():
            dense_probs = torch.zeros((self.num_var, self.max_dom), dtype=torch.float32, device=self.device)
            dense_probs[self.var_idx, self.dom_idx] = value_prob.view(-1)

            idx = torch.multinomial(dense_probs, 1)
            idx += self.var_off.view(-1, 1)

            assignment = torch.zeros((self.num_val, 1), dtype=torch.float32, device=self.device)
            assignment[idx.view(-1)] = 1.0
        sampled_prob = value_prob[idx]
        log_prob = scatter_sum(torch.log(sampled_prob + 1.0e-5), self.batch, dim=0).view(-1, 1)
        return assignment, log_prob

    def hard_assign_sample_local(self, logits, assignment):
        value_prob = self.value_softmax_local(logits, assignment).view(self.num_val, 1)
        with torch.no_grad():
            value_assignment = self.dom_idx[assignment.bool().flatten()]

            dense_probs = torch.zeros((self.batch_size, self.max_num_val), dtype=torch.float32, device=self.device)
            dense_probs[self.batch[self.var_idx], self.batch_val_idx] = value_prob.view(-1)

            idx = torch.multinomial(dense_probs, 1).flatten()
            idx += self.batch_val_off

            value_assignment[self.var_idx[idx]] = self.dom_idx[idx]
            assignment = torch.zeros((self.num_val, 1), dtype=torch.float32, device=self.device)
            assignment[self.var_off + value_assignment] = 1.0

        log_prob = torch.log(value_prob[idx] + 1.0e-5).view(-1, 1)
        return assignment, log_prob

    def constraint_is_sat(self, assignment_one_hot, update_LE=False):
        assignment_one_hot = assignment_one_hot.view(self.num_val, -1)
        sat = torch.cat([c.is_sat(assignment_one_hot, update_LE) for k, c in self.constraints.items()], dim=0)
        if update_LE:
            self.update_LE()
        return sat>0

    def count_unsat(self, assignment_one_hot):
        unsat = 1.0 - self.constraint_is_sat(assignment_one_hot).float()
        num_unsat = scatter_sum(unsat, self.cst_batch, dim=0, dim_size=self.batch_size)
        return num_unsat

    def count_sat(self, assignment_one_hot):
        sat = self.constraint_is_sat(assignment_one_hot).float()
        sat = scatter_sum(sat, self.cst_batch, dim=0, dim_size=self.batch_size)
        return sat

    def constraint_min_sat(self, assignment_one_hot, update_LE=False):
        assignment_one_hot = assignment_one_hot.view(self.num_val, -1)
        sat = self.constraints['ext'].is_sat(assignment_one_hot, update_LE)
        # sat = torch.cat([c.is_sat(assignment_one_hot, update_LE) for k, c in self.constraints.items()], dim=0)
        sat, _ = scatter_min(sat, self.cst_batch, dim=0, dim_size=self.batch_size)

        if update_LE:
            self.update_LE()
        return sat