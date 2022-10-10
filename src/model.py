import torch
from sklearn.metrics import roc_auc_score, average_precision_score
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GAE, GCNConv

from helpers import to_probs
from proxssi.groups.gcn import gcn_groups
from proxssi.groups.linear import linear_groups
from proxssi.optimizers.adamw_hf import AdamW


class MyGAE(GAE):
    def test(self, z, pos_edge_index, neg_edge_index):
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred), y, pred


class GCNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)


class LinearEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LinearEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class ConceptGAE(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, output_dim, linear=False):
        super(ConceptGAE, self).__init__()
        self.mode = mode
        if self.mode == 'embs':
            self.mfs_weights = torch.nn.Parameter(torch.randn(1000, 5))
        elif self.mode == 'joint':
            self.mfs_weights = torch.nn.Parameter(torch.randn(1000, 5))
            self.af_weights = torch.nn.Parameter(torch.randn(1000))
        if linear:
            self.gae = MyGAE(LinearEncoder(input_dim, hidden_dim, output_dim))
        else:
            self.gae = MyGAE(GCNEncoder(input_dim, hidden_dim, output_dim))

    def forward(self, x, train_pos_edge_index):
        if self.mode == 'embs':
            x_temp = torch.zeros(x.size(0), 1000).to(x.device)
            for i in range(1000):
                c = i * 5
                c_weighted = x[:, c:c + 5] * to_probs(self.mfs_weights[i])
                x_temp[:, i] = c_weighted.sum(dim=1)
            x = x_temp
        elif self.mode == 'joint':
            x_1, x_2 = x[:, :1000], x[:, 1000:]
            x_2_temp = torch.zeros_like(x_1)
            for i in range(x_2_temp.size(1)):
                c = i * 5
                c_weighted = x[:, c:c + 5] * to_probs(self.mfs_weights[i])
                x_2_temp[:, i] = c_weighted.sum(dim=1)
            x_2 = x_2_temp
            alphas = torch.clamp(self.af_weights, min=0, max=1)
            x = alphas * x_1 + (1 - alphas) * x_2
        z = self.gae.encode(x, train_pos_edge_index)
        return z


class SparseConceptGAE(nn.Module):
    def __init__(self, mode, input_dim, hidden_dim, output_dim, lr, lambda_r, linear=False):
        super(SparseConceptGAE, self).__init__()
        self.model = ConceptGAE(mode, input_dim, hidden_dim, output_dim, linear)
        if linear:
            grouped_params = linear_groups(self.model, weight_decay=0)
        else:
            grouped_params = gcn_groups(self.model, weight_decay=0)
        optimizer_kwargs = {
            'lr': lr,
            'penalty': 'l1_l2',
            'prox_kwargs': {'lambda_': lambda_r}
        }
        self.optimizer = AdamW(grouped_params, **optimizer_kwargs)

    def train(self, x, train_pos_edge_index):
        self.model.train()
        self.optimizer.zero_grad()
        z = self.model(x, train_pos_edge_index)
        loss = self.model.gae.recon_loss(z, train_pos_edge_index)
        loss.backward()
        self.optimizer.step()

    def test(self, x, train_pos_edge_index, test_pos_edge_index, test_neg_edge_index):
        self.model.eval()
        with torch.no_grad():
            z = self.model(x, train_pos_edge_index)
            auc, ap, y, pred = self.model.gae.test(z, test_pos_edge_index, test_neg_edge_index)
            return auc, ap, y, pred
