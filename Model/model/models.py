import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from numpy.random import RandomState


def get_param(shape):
    param = nn.Parameter(torch.Tensor(*shape))
    nn.init.xavier_normal_(param.data)
    return param

def get_linear(in_dim, out_dim, drop=0.3):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim),
        nn.Tanh(),
        nn.Dropout(drop)
    )

class ConvE(nn.Module):
    def __init__(self, num_ents, num_rels, params):
        super(self.__class__, self).__init__()
        self.in_dim, self.out_dim = 100, 200
        self.out_channel = 100
        self.sz_h, self.sz_w = 20, 10
        self.ker_size = 7
        self.num_ents, self.num_rels = num_ents, num_rels
        self.p = params

        self.flat_h = int(2 * self.sz_h) - self.ker_size + 1
        self.flat_w = self.sz_w - self.ker_size + 1
        self.flat_sz = self.flat_h * self.flat_w * self.out_channel

        self.E = get_param((num_ents, self.out_dim))
        self.R = get_param((num_rels * 2, self.out_dim))
        self.bias = nn.Parameter(torch.zeros(num_ents))

        self.loss = torch.nn.BCELoss()
        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channel)
        self.bn2 = torch.nn.BatchNorm1d(self.out_dim)

        self.hidden_drop = torch.nn.Dropout(self.p.drop1)
        self.hidden_drop2 = torch.nn.Dropout(self.p.drop2)
        self.feature_drop = torch.nn.Dropout(self.p.drop3)
        self.m_conv1 = torch.nn.Conv2d(1, self.out_channel, kernel_size=(self.ker_size, self.ker_size),
                                       stride=1, padding=0, bias=True)
        self.fc = torch.nn.Linear(self.flat_sz, self.out_dim)

    def concat(self, e1_embed, rel_embed):
        e1_embed = e1_embed.view(-1, 1, self.out_dim)  # [B, 1, F1]
        rel_embed = rel_embed.view(-1, 1, self.out_dim)  # [B, 1, F1]
        stack_inp = torch.cat([e1_embed, rel_embed], 1)  # [B, 2, F1]
        stack_inp = torch.transpose(stack_inp, 2, 1).reshape(
            (-1, 1, 2 * self.sz_h, self.sz_w))  # [B, 1, 2*C1, C2]
        return stack_inp

    def forward(self, e1_idx, r_idx):  # B, B
        X = self.E
        sub_emb, all_ent = X[e1_idx], X
        rel_emb = self.R[r_idx]

        stk_inp = self.concat(sub_emb, rel_emb)
        x = self.bn0(stk_inp)
        x = self.m_conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.feature_drop(x)
        x = x.view(-1, self.flat_sz)
        x = self.fc(x)
        x = self.hidden_drop2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = torch.mm(x, all_ent.transpose(1, 0))
        x += self.bias.expand_as(x)

        score = torch.sigmoid(x)
        return score

class LowFER(nn.Module):
    def __init__(self, num_ents, num_rels, params):
        super(LowFER, self).__init__()
        self.in_dim, self.out_dim = 100, 200
        self.p = params

        self.E = get_param((num_ents, self.out_dim))
        self.R = get_param((num_rels * 2, self.p.rdim))
        self.k = self.p.k
        self.U = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.out_dim, self.k * self.out_dim)),
                                           dtype=torch.float, requires_grad=True))
        self.V = nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.p.rdim, self.k * self.out_dim)),
                                           dtype=torch.float, requires_grad=True))
        self.input_dropout = nn.Dropout(self.p.drop1)
        self.hidden_dropout1 = nn.Dropout(self.p.drop2)
        self.hidden_dropout2 = nn.Dropout(self.p.drop3)
        self.bn0 = nn.BatchNorm1d(self.out_dim)
        self.bn1 = nn.BatchNorm1d(self.out_dim)
        self.loss = nn.BCELoss()

    def forward(self, e1_idx, r_idx):
        X = self.E
        e1, x_t = X[e1_idx], X
        r = self.R[r_idx]

        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        # MFB
        x = torch.mm(e1, self.U) * torch.mm(r, self.V)
        x = self.hidden_dropout1(x)
        x = x.view(-1, self.out_dim, self.k)  # [B, d1, k]
        x = x.sum(-1)  # [B, d1]
        x = torch.mul(torch.sign(x), torch.sqrt(torch.abs(x) + 1e-12))
        x = F.normalize(x, p=2, dim=-1)
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = x @ x_t.t()

        pred = torch.sigmoid(x)
        return pred

class HypER(nn.Module):
    def __init__(self, num_ents, num_rels, params):
        super(HypER, self).__init__()
        self.out_channels = 100
        self.filt_h, self.filt_w = 1, 9
        self.in_dim, self.out_dim = 100, 200
        self.p = params

        self.E = get_param((num_ents, self.out_dim))
        self.R = get_param((num_rels * 2, self.out_dim))
        self.input_drop = torch.nn.Dropout(self.p.drop1)
        self.hidden_drop = torch.nn.Dropout(self.p.drop2)
        self.feature_map_drop = torch.nn.Dropout2d(self.p.drop3)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm2d(1)
        self.bn1 = torch.nn.BatchNorm2d(self.out_channels)
        self.bn2 = torch.nn.BatchNorm1d(self.out_dim)
        self.register_parameter('b', nn.Parameter(torch.zeros(num_ents)))
        fc_length = (1 - self.filt_h + 1) * (self.out_dim - self.filt_w + 1) * self.out_channels
        self.fc = torch.nn.Linear(fc_length, self.out_dim)
        fc1_length = self.out_channels * self.filt_h * self.filt_w
        self.fc1 = torch.nn.Linear(self.out_dim, fc1_length)

    def forward(self, e1_idx, r_idx):
        X = self.E
        e1, x_t = X[e1_idx].view(-1, 1, 1, self.out_dim), X
        r = self.R[r_idx]

        e1 = self.bn0(e1)
        e1 = self.input_drop(e1)

        k = self.fc1(r)
        k = k.view(e1.size(0) * self.out_channels, 1, self.filt_h, self.filt_w)  # [B*C, 1, 1, K]
        x = e1.permute(1, 0, 2, 3)  # [1, B, 1, F]

        x = F.conv2d(x, k, groups=e1_idx.size(0))
        x = x.view(e1_idx.size(0), 1, self.out_channels, 1 - self.filt_h + 1, e1.size(3) - self.filt_w + 1)
        x = x.permute(0, 3, 4, 1, 2)
        x = torch.sum(x, dim=3)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = self.bn1(x)
        x = self.feature_map_drop(x)
        x = x.view(e1_idx.size(0), -1)
        x = self.fc(x)
        x = self.hidden_drop(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = torch.mm(x, x_t.transpose(1, 0))
        x += self.b.expand_as(x)
        pred = torch.sigmoid(x)
        return pred


class TuckER(nn.Module):
    def __init__(self, num_ents, num_rels, params=None):
        super(TuckER, self).__init__()
        self.in_dim, self.out_dim = 100, 200
        self.p = params

        self.E = get_param((num_ents, self.out_dim))
        self.R = get_param((num_rels * 2, self.p.rdim))
        self.W = torch.nn.Parameter(torch.tensor(np.random.uniform(-1, 1, (self.p.rdim, self.out_dim * self.out_dim)),
                                                 dtype=torch.float, requires_grad=True))

        self.input_dropout = torch.nn.Dropout(self.p.drop1)
        self.hidden_dropout1 = torch.nn.Dropout(self.p.drop2)
        self.hidden_dropout2 = torch.nn.Dropout(self.p.drop3)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(self.out_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.out_dim)

    def forward(self, e1_idx, r_idx):
        X = self.E
        e1, x_t = X[e1_idx], X
        r = self.R[r_idx]

        e1 = self.bn0(e1)
        e1 = self.input_dropout(e1)
        x = e1.view(-1, 1, e1.size(1))

        W_mat = r @ self.W  # [B, F*F]
        W_mat = W_mat.view(-1, e1.size(1), e1.size(1))  # [B, F, F]
        W_mat = self.hidden_dropout1(W_mat)

        x = torch.bmm(x, W_mat)  # [B, 1, F] [B, F, F] -> [B, 1, F]
        x = x.view(-1, e1.size(1))
        x = self.bn1(x)
        x = self.hidden_dropout2(x)
        x = torch.mm(x, x_t.transpose(1, 0))
        pred = torch.sigmoid(x)
        return pred

class HeLpER(nn.Module):
    def __init__(self, num_ents, num_rels, params):
        super(HeLpER, self).__init__()
        self.p = params
        self.in_channels, self.out_channels = self.p.in_ch, self.p.out_ch
        self.ker = 9
        self.stride = 1
        self.num_ents, self.num_rels = num_ents, num_rels
        self.in_dim, self.out_dim = 200, self.p.k * 200

        self.E = get_param((num_ents, self.in_dim))
        self.R = get_param((num_rels * 2, self.out_channels, self.in_channels, self.ker))
        self.after_conv_len = (self.out_dim//self.in_channels-self.ker)//self.stride+1
        self.fc_length = self.out_channels*self.after_conv_len
        self.e_W = get_param((self.in_dim, self.out_dim))

        self.input_dropout = nn.Dropout(self.p.drop1)
        self.hidden_drop = torch.nn.Dropout(self.p.drop2)
        self.loss = torch.nn.BCELoss()

        self.bn0 = torch.nn.BatchNorm1d(self.in_dim)
        self.bn1 = torch.nn.BatchNorm1d(self.out_channels)
        self.out_layer = get_linear(self.fc_length, self.in_dim, drop=self.p.drop3)

    def forward(self, e1_idx, r_idx):
        x, X = self.E[e1_idx], self.E
        x = self.bn0(x)
        x = self.input_dropout(x)
        x = x @ self.e_W
        k = self.R[r_idx]

        x = x.reshape(1, e1_idx.shape[0] * self.in_channels, self.out_dim//self.in_channels)
        k = k.reshape(e1_idx.shape[0] * self.out_channels, self.in_channels, self.ker)  # [B*C, 1, K]

        x = F.conv1d(x, k, groups=e1_idx.shape[0])  # [B*C, F]
        x = x.reshape(e1_idx.shape[0], self.out_channels, self.after_conv_len)
        x = self.bn1(x)
        x = self.hidden_drop(x)
        x = x.reshape(e1_idx.shape[0], self.fc_length)

        x = self.out_layer(x)
        x = x @ X.t()
        pred = torch.sigmoid(x)
        return pred

# class HeLpER(nn.Module):
#     def __init__(self, num_ents, num_rels, params):
#         super(HeLpER, self).__init__()
#         self.p = params
#         self.in_channels, self.out_channels = self.p.in_ch, self.p.out_ch
#         self.ker = 9
#         self.stride = 1
#         self.pool = 3
#         self.num_ents, self.num_rels = num_ents, num_rels
#         # self.in_dim, self.out_dim = 200, self.p.k * 200
#         # self.in_dim, self.out_dim = 339, 339
#         self.in_dim, self.out_dim = 200, 200
#
#         self.E = get_param((num_ents, self.in_dim))
#         self.R = get_param((num_rels * 2, self.out_channels, self.in_channels, self.ker))
#         # self.R = get_param((num_rels * 2, 20))
#         # self.R = get_param((num_rels * 2, 435))
#         # self.R = get_param((num_rels * 2, 55))
#         self.after_conv_len = (self.out_dim//self.in_channels-self.ker)//self.stride+1
#         self.fc_length = self.out_channels*self.after_conv_len
#         # self.e_W = get_param((self.in_dim, self.out_dim))
#         # self.r_W = get_param((20,  self.out_channels * self.in_channels * self.ker))
#         # self.r_W = get_param((435,  self.out_channels * self.in_channels * self.ker))
#         # self.r_W = get_param((55,  self.out_channels * self.in_channels * self.ker))
#
#         self.input_dropout = nn.Dropout(self.p.drop1)
#         self.hidden_drop = torch.nn.Dropout(self.p.drop2)
#         self.out_drop = torch.nn.Dropout(self.p.drop3)
#         self.loss = torch.nn.BCELoss()
#
#         self.bn0 = torch.nn.BatchNorm1d(self.in_dim)
#         # self.bn0 = torch.nn.BatchNorm1d(self.out_dim)
#         self.bn1 = torch.nn.BatchNorm1d(self.out_channels)
#         # self.bn1 = torch.nn.BatchNorm1d(self.out_dim)
#         # self.bn2 = torch.nn.BatchNorm1d(self.in_dim)
#         # self.bn2 = torch.nn.BatchNorm1d(self.out_dim)
#         # self.bn1 = torch.nn.BatchNorm1d(self.p.k*self.out_channels//self.in_channels//self.pool)
#         self.out_layer = get_linear(self.fc_length, self.in_dim, drop=self.p.drop3)
#
#     def forward(self, e1_idx, r_idx):
#         x, X = self.E[e1_idx], self.E
#         # X = self.E @ self.e_W
#         # x = X[e1_idx]
#         x = self.bn0(x)
#         x = self.input_dropout(x)
#         # x = x @ self.e_W
#         k = self.R[r_idx]
#         # k = self.R[r_idx] @ self.r_W
#
#         x = x.reshape(1, e1_idx.shape[0] * self.in_channels, self.out_dim//self.in_channels)
#         # x = x.reshape(1, e1_idx.shape[0] * self.in_channels, self.pool, self.out_dim//self.in_channels//self.pool).sum(dim=2)
#         k = k.reshape(e1_idx.shape[0] * self.out_channels, self.in_channels, self.ker)  # [B*C, 1, K]
#
#         x = F.conv1d(x, k, groups=e1_idx.shape[0])  # [B*C, F]
#         # x = F.conv1d(x, k, groups=e1_idx.shape[0], padding='same')  # [B*C, F]
#         x = x.reshape(e1_idx.shape[0], self.out_channels, self.after_conv_len)
#         # x = x.reshape(e1_idx.shape[0], self.out_dim)
#         # x = x.reshape(e1_idx.shape[0], self.out_channels*self.p.k//self.in_channels//self.pool, self.in_dim)
#         # x = x.reshape(e1_idx.shape[0], self.out_channels*self.p.k//self.in_channels, self.in_dim)
#         x = self.bn1(x)
#         x = self.hidden_drop(x)
#         x = x.reshape(e1_idx.shape[0], self.fc_length)
#         # x = x.sum(dim=1)
#
#         x = self.out_layer(x)
#         # x = self.out_drop(x)
#         # x = torch.tanh(x)
#         pred = torch.sigmoid(x @ X.t())
#         return pred