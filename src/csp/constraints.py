import torch
from time import time
from torch_geometric.utils import degree
from torch_scatter import scatter_sum, scatter_min, scatter_max, segment_max_csr


class Constraint_Data_Base:

    def __init__(self, cq_data, cst_edges, batch=None):
        self.cst_edges = cst_edges

        self.LE = None
        self.num_cst = cst_edges[0].max().cpu().numpy() + 1
        self.num_edges = cst_edges.shape[1]

        # Single-link predictor
        self.predictor = cq_data.predictor

        # The degree of each constraint node
        self.cst_deg = degree(cst_edges[0], dtype=torch.int64)
        # The constraint-degree of each value node
        self.val_ids = cq_data.val_ids
        self.val_deg = degree(cst_edges[1], dtype=torch.int64, num_nodes=cq_data.num_val)

        self.var_idx = cq_data.var_idx

        self.batch = batch if batch is not None else torch.zeros((self.num_cst,), dtype=torch.int64)
        self.batch_size = int(self.batch.max().cpu().numpy())

        self.device = cq_data.device

    def to(self, device):
        self.device = device
        self.cst_edges = self.cst_edges.to(device)
        if not self.LE is None:
            self.LE = self.LE.to(device)
        self.predictor = self.predictor.to(device)
        self.cst_deg = self.cst_deg.to(device)
        self.val_ids = self.val_ids.to(device)
        self.val_deg = self.val_deg.to(device)
        self.var_idx = self.var_idx.to(device)
        self.batch = self.batch.to(device)

    def update_LE_(self, **kwargs):
        raise NotImplementedError

    def is_sat(self, assignment):
        raise NotImplementedError



class CQ_Constraint_Data(Constraint_Data_Base):

    def __init__(self, cq_data, cst_type, dis_part, cst_edges, head_mask, rel_ids=None, head_ids=None, tail_ids=None, PE=None, batch=None):

        self.cst_type = cst_type

        self.head_ids = head_ids
        self.rel_ids = rel_ids
        self.tail_ids = tail_ids

        self.dis_part = dis_part
        
        self.head_mask = head_mask
        self.PE = PE

        super(CQ_Constraint_Data, self).__init__(
                cq_data=cq_data,
                cst_edges=cst_edges,
                batch=batch,
        )

        self.cst_neg_mask = self.cst_type.bool()
        self.neg_edge_mask = self.cst_neg_mask[self.cst_edges[0]]

        self.to(cq_data.device)

    def to(self, device):
        super(CQ_Constraint_Data, self).to(device)
        self.cst_type = self.cst_type.to(device)
        self.dis_part = self.dis_part.to(device)
        self.head_ids = self.head_ids.to(device)
        self.rel_ids = self.rel_ids.to(device)
        self.tail_ids = self.tail_ids.to(device)
        self.head_mask = self.head_mask.to(device)
        self.PE = self.PE.to(device)
        
        self.cst_neg_mask = self.cst_neg_mask.to(device)
        self.neg_edge_mask = self.neg_edge_mask.to(device)
        
    @staticmethod
    def collate(batch_list, merged_cq_data):
        cst_type, dis_part, batch_idx = [], [], []
        head_ids, rel_ids, tail_ids, head_mask = [], [], [], []
        PEs = []

        cst_off = 0
        cst_edges = []


        for cst_data, var_off, val_off, i in batch_list:

            new_dis_part = cst_data.dis_part
            if len(dis_part)>0:
                new_dis_part[new_dis_part>0] += dis_part[-1].max()
            dis_part.append(new_dis_part) 

            cst_type.append(cst_data.cst_type)
            batch_idx.append(cst_data.batch + i)

            head_ids.append(cst_data.head_ids + var_off)
            tail_ids.append(cst_data.tail_ids + var_off)
            rel_ids.append(cst_data.rel_ids)
            head_mask.append(cst_data.head_mask)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_edges.append(cur_cst_val_edges)

            PEs.append(cst_data.PE)

            cst_off += cst_data.num_cst

        cst_type = torch.cat(cst_type, dim=0)
        dis_part = torch.cat(dis_part, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        
        head_ids = torch.cat(head_ids, dim=0)
        tail_ids = torch.cat(tail_ids, dim=0)
        rel_ids = torch.cat(rel_ids, dim=0)
        head_mask = torch.cat(head_mask, dim=0)
        
        cst_edges = torch.cat(cst_edges, dim=1)

        PEs = torch.cat(PEs, dim=0).long()
        
        batch_cst_data = CQ_Constraint_Data(
            cq_data=merged_cq_data,
            cst_type=cst_type,
            dis_part=dis_part,
            cst_edges=cst_edges,
            head_mask = head_mask,
            rel_ids = rel_ids,
            head_ids = head_ids,
            tail_ids = tail_ids,
            PE = PEs,
            batch=batch_idx
        )
        return batch_cst_data

    def update_LE_(self, assignment, **kwargs):
        
        # Evaluate the value assignments for each value
        val_assignment, _ = scatter_max(assignment.flatten()*self.val_ids, self.var_idx)

        # Generate the head/tail scores for all constraints, under the current assignment
        # 7s / 150 steps
        with torch.no_grad():
            head_scores = self.predictor.head_scores(val_assignment[self.tail_ids], self.rel_ids)
            tail_scores = self.predictor.tail_scores(val_assignment[self.head_ids], self.rel_ids)

        # Take the head and tail score for each edge
        edge_head_scores = head_scores[self.cst_edges[0], self.val_ids[self.cst_edges[1]]]
        edge_tail_scores = tail_scores[self.cst_edges[0], self.val_ids[self.cst_edges[1]]]

        # Evaluate new LE
        self.LE = (edge_head_scores*self.head_mask + edge_tail_scores*(~self.head_mask) > 0).long()
        self.LE[self.cst_type[self.cst_edges[0]] == 1] = 1 - self.LE[self.cst_type[self.cst_edges[0]]==1]
        

    def is_sat(self, assignment, update_val_comp=False):

        # Evaluate the value assignments for each variable
        val_assignment, _ = scatter_max(assignment.flatten()*self.val_ids, self.var_idx)

        # Gather the values of heads and tails of constraints
        head_vals = val_assignment[self.head_ids].long()
        tail_vals = val_assignment[self.tail_ids].long()
        
        # Predict the satisfiability scores
        # 6s / 150 steps
        with torch.no_grad():
            scores = self.predictor(head_vals, self.rel_ids, tail_vals)

        # Consider negations
        scores = scores * (1-2*self.cst_type)

        # Consider the disjunctions
        dis_scores, _ = scatter_max(scores, self.dis_part)
        scores[self.dis_part > 0] = dis_scores[self.dis_part][self.dis_part>0]

        # Update LE if necessary
        # 6s / 150 steps
        if update_val_comp:
            self.update_LE_(assignment)
        
        return scores