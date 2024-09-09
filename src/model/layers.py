import torch
from torch.nn import Module, Linear, Sequential, LayerNorm, ReLU
from torch_geometric.utils import scatter


# general aggregation method with/without sparse_coo installation
def aggregate(msg, out_idx, in_idx, dim_size, aggr):
    return scatter(msg[out_idx], in_idx, dim=0, dim_size=dim_size, reduce=aggr )

class Val2Cst_Layer(Module):

    def __init__(self, config, use_PE=True):
        super(Val2Cst_Layer, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.aggr = config['aggr_val2cst']
        self.use_PE = use_PE
        assert self.aggr in ['sum', 'mean', 'max']

        # E
        self.val_enc = Sequential(
            Linear(self.hidden_dim + 1, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim, bias=False),
            LayerNorm(self.hidden_dim),
        )

        # M_V
        self.val_send = Sequential(
            Linear(self.hidden_dim, 4 * self.hidden_dim, bias=False),
            LayerNorm(4 * self.hidden_dim),
        )

    def forward(self, data, h_val, assign):
        # Encode value state
        x_val = torch.cat([h_val, assign.view(-1, 1).half()], dim=1)
        x_val = self.val_enc(x_val)

        # Values generate msg.
        m_val = self.val_send(x_val)
        m_val = m_val.view(4 * data.num_val, self.hidden_dim)

        # Edge index from edge labels
        out_idx = 4 * data.cst_edges[1] + 2 * data.LE + (data.PE if self.use_PE else 0*data.PE)
        in_idx = data.cst_edges[0]

        # Aggregate
        r_cst = aggregate(m_val, out_idx, in_idx, data.num_cst, self.aggr)
        return r_cst, x_val


class Cst2Val_Layer(Module):

    def __init__(self, config, use_PE = True):
        super(Cst2Val_Layer, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.aggr = config['aggr_cst2val']
        self.use_PE = use_PE
        assert self.aggr in ['sum', 'mean', 'max']

        # M_C
        self.cst_send = Sequential(
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, 4 * self.hidden_dim, bias=False),
            LayerNorm(4 * self.hidden_dim),
        )

        # U_V
        self.val_rec = Sequential(
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim, bias=False),
            LayerNorm(self.hidden_dim)
        )

    def forward(self, data, x_val, r_cst):
        # Constraints generate msg
        m_cst = self.cst_send(r_cst)
        m_cst = m_cst.view(4 * data.num_cst, self.hidden_dim)

        # Edge index from edge labels
        out_idx = 4 * data.cst_edges[0] + 2 * data.LE + (data.PE if self.use_PE else 0*data.PE)
        in_idx = data.cst_edges[1]

        # Aggregate and update
        r_val = aggregate(m_cst, out_idx, in_idx, data.num_val, self.aggr)
        x_val = self.val_rec(x_val + r_val) + x_val
        return x_val


class Val2Val_Layer(Module):

    def __init__(self, config):
        super(Val2Val_Layer, self).__init__()
        self.hidden_dim = config['hidden_dim']
        self.aggr = config['aggr_val2var']

        assert self.aggr in ['sum', 'mean', 'max']

        # U_X
        self.var_enc = Sequential(
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, self.hidden_dim, bias=False),
            LayerNorm(self.hidden_dim),
        )

    def forward(self, data, y_val):
        # Pool value states of the same variable
        if self.aggr == 'max':
            z_var = scatter(y_val, data.var_idx, dim=0, dim_size=data.num_var, reduce='max')
        elif self.aggr == 'mean':
            z_var = scatter(y_val, data.var_idx, dim=0, dim_size=data.num_var, reduce = 'mean')
        else:
            z_var = scatter(y_val, data.var_idx, dim=0, dim_size=data.num_var, reduce = 'sum')

        # Apply U_X and send result back
        z_var = self.var_enc(z_var)
        y_val += z_var[data.var_idx]
        return y_val


class Policy(Module):

    def __init__(self, config):
        super(Policy, self).__init__()
        self.hidden_dim = config['hidden_dim']

        # O
        self.mlp = Sequential(
            LayerNorm(self.hidden_dim),
            Linear(self.hidden_dim, self.hidden_dim, bias=True),
            ReLU(),
            Linear(self.hidden_dim, 1, bias=False),
        )

    def forward(self, h_val):
        logits = self.mlp(h_val)
        return logits
