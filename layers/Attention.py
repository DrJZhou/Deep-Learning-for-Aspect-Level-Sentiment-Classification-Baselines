import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, embed_dim, hidden_dim=None, out_dim=None, n_head=1, score_function='scaled_dot_product',
                 dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(Attention, self).__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim // n_head
        if out_dim is None:
            out_dim = embed_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim, hidden_dim))
        self.proj = nn.Linear(n_head * hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim * 2))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        self.w_kx.data.uniform_(-stdv, stdv)
        self.w_qx.data.uniform_(-stdv, stdv)
        if self.weight is not None:
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output


class AttentionAspect(nn.Module):
    def __init__(self, embed_dim_k, embed_dim_q, hidden_dim_k=None, hidden_dim_q=None, out_dim=None, n_head=1, score_function='scaled_dot_product',
                 dropout=0):
        ''' Attention Mechanism
        :param embed_dim:
        :param hidden_dim:
        :param out_dim:
        :param n_head: num of head (Multi-Head Attention)
        :param score_function: scaled_dot_product / mlp (concat) / bi_linear (general dot)
        :return (?, q_len, out_dim,)
        '''
        super(AttentionAspect, self).__init__()
        if hidden_dim_k is None:
            hidden_dim_k = embed_dim_k // n_head
        if hidden_dim_q is None:
            hidden_dim_q = embed_dim_q // n_head
        if out_dim is None:
            out_dim = embed_dim_k
        self.embed_dim_k = embed_dim_k
        self.embed_dim_q = embed_dim_q
        self.hidden_dim_k = hidden_dim_k
        self.hidden_dim_q = hidden_dim_q
        self.n_head = n_head
        self.score_function = score_function
        self.w_kx = nn.Parameter(torch.FloatTensor(n_head, embed_dim_k, hidden_dim_k))
        self.w_qx = nn.Parameter(torch.FloatTensor(n_head, embed_dim_q, hidden_dim_q))
        self.proj = nn.Linear(n_head * hidden_dim_k, out_dim)
        self.dropout = nn.Dropout(dropout)
        if score_function == 'mlp':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim_k + hidden_dim_q))
        elif self.score_function == 'bi_linear':
            self.weight = nn.Parameter(torch.Tensor(hidden_dim_q, hidden_dim_k))
        else:
            self.register_parameter('weight', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv_k = 1. / math.sqrt(self.hidden_dim_k)
        self.w_kx.data.uniform_(-stdv_k, stdv_k)
        stdv_q = 1. / math.sqrt(self.hidden_dim_q)
        self.w_qx.data.uniform_(-stdv_q, stdv_q)
        if self.weight is not None:
            stdv = 1. / math.sqrt((self.hidden_dim_k+self.hidden_dim_q)/2)
            self.weight.data.uniform_(-stdv, stdv)

    def forward(self, k, q):
        if len(q.shape) == 2:  # q_len missing
            q = torch.unsqueeze(q, dim=1)
        if len(k.shape) == 2:  # k_len missing
            k = torch.unsqueeze(k, dim=1)
        mb_size = k.shape[0]  # ?
        k_len = k.shape[1]
        q_len = q.shape[1]
        # k: (?, k_len, embed_dim,)
        # q: (?, q_len, embed_dim,)
        # kx: (n_head, ?*k_len, embed_dim) -> (n_head*?, k_len, hidden_dim)
        # qx: (n_head, ?*q_len, embed_dim) -> (n_head*?, q_len, hidden_dim)
        # score: (n_head*?, q_len, k_len,)
        # output: (?, q_len, out_dim,)
        kx = k.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim_k)  # (n_head, ?*k_len, embed_dim)
        qx = q.repeat(self.n_head, 1, 1).view(self.n_head, -1, self.embed_dim_q)  # (n_head, ?*q_len, embed_dim)
        kx = torch.bmm(kx, self.w_kx).view(-1, k_len, self.hidden_dim_k)  # (n_head*?, k_len, hidden_dim)
        qx = torch.bmm(qx, self.w_qx).view(-1, q_len, self.hidden_dim_q)  # (n_head*?, q_len, hidden_dim)
        if self.score_function == 'scaled_dot_product':
            kt = kx.permute(0, 2, 1)
            qkt = torch.bmm(qx, kt)
            score = torch.div(qkt, math.sqrt(self.hidden_dim_k))
        elif self.score_function == 'mlp':
            kxx = torch.unsqueeze(kx, dim=1).expand(-1, q_len, -1, -1)
            qxx = torch.unsqueeze(qx, dim=2).expand(-1, -1, k_len, -1)
            kq = torch.cat((kxx, qxx), dim=-1)  # (n_head*?, q_len, k_len, hidden_dim*2)
            score = F.tanh(torch.matmul(kq, self.weight))
        elif self.score_function == 'bi_linear':
            qw = torch.matmul(qx, self.weight)
            kt = kx.permute(0, 2, 1)
            score = torch.bmm(qw, kt)
        else:
            raise RuntimeError('invalid score_function')
        score = F.softmax(score, dim=-1)
        output = torch.bmm(score, kx)  # (n_head*?, q_len, hidden_dim)
        output = torch.cat(torch.split(output, mb_size, dim=0), dim=-1)  # (?, q_len, n_head*hidden_dim)
        output = self.proj(output)  # (?, q_len, out_dim)
        output = self.dropout(output)
        return output, score



class BasicAttention(nn.Module):
    def __init__(self, hidden_dim=None, score_function='basic'):
        super(BasicAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.score_function = score_function
        if score_function == "basic":
            self.w = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
            self.b = nn.Parameter(torch.Tensor(hidden_dim))
            self.u = nn.Parameter(torch.Tensor(hidden_dim, 1))
        if score_function == 'aspect':
            self.w = nn.Parameter(torch.Tensor(hidden_dim, hidden_dim))
        if score_function == 'simple':
            self.w = nn.Parameter(torch.Tensor(hidden_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_dim)
        if self.score_function == "basic":
            self.w.data.uniform_(-stdv, stdv)
            self.u.data.uniform_(-stdv, stdv)
        elif self.score_function == "aspect":
            self.w.data.uniform_(-stdv, stdv)
        elif self.score_function == "simple":
            self.w.data.uniform_(-stdv, stdv)

    def forward(self, k, q=None):
        if self.score_function == "basic":
            # after this, we have (batch, dim1) with a diff weight per each cell
            attention_score = torch.matmul(k, self.w) + self.b
            attention_score = torch.tanh(attention_score)
            attention_score = torch.matmul(attention_score, self.u).squeeze()
        elif self.score_function == "aspect":
            if len(q.shape) == 2:  # q_len missing
                q = torch.unsqueeze(q, dim=2)
            attention_score = torch.matmul(k, self.w)
            attention_score = torch.bmm(attention_score, q).squeeze()
        elif self.score_function == "simple":
            attention_score = torch.matmul(k, self.w).squeeze()
        attention_score = F.softmax(attention_score).view(k.size(0), k.size(1), 1)
        scored_x = k * attention_score
        # now, sum across dim 1 to get the expected feature vector
        condensed_x = torch.sum(scored_x, dim=1)
        return condensed_x, attention_score
