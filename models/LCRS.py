from layers.Dynamic_RNN import DynamicRNN
from layers.Attention import Attention
import torch
import torch.nn as nn
import torch.nn.functional as F


class LCRS(nn.Module):
    def __init__(self, args, embedding_matrix, aspect_embedding_matrix=None, memory_weighter='no'):
        super(LCRS, self).__init__()
        self.args = args
        self.memory_weighter = memory_weighter
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.blstm_l = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, dropout=args.dropout,
                                  rnn_type='LSTM')
        self.blstm_c = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, dropout=args.dropout,
                                  rnn_type='LSTM')
        self.blstm_r = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, dropout=args.dropout,
                                  rnn_type='LSTM')
        self.dense = nn.Linear(args.hidden_dim * 4, args.polarities_dim)
        # target to context attention
        self.t2c_l_attention = Attention(args.hidden_dim, score_function='bi_linear')
        self.t2c_r_attention = Attention(args.hidden_dim, score_function='bi_linear')
        # context to target attention
        self.c2t_l_attention = Attention(args.hidden_dim, score_function='bi_linear')
        self.c2t_r_attention = Attention(args.hidden_dim, score_function='bi_linear')
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def locationed_memory(self, memory_l, memory_r, x_l_len, x_c_len, x_r_len, dep_offset=None):
        position = 'linear'
        memory_len = x_l_len + x_c_len + x_r_len
        # loop over samples in the batch
        for i in range(memory_l.size(0)):
            # Loop over memory slices
            for idx in range(memory_len[i]):
                aspect_start = x_l_len[i] + 1  # INCORRECT: ASSUME x_l_len = 0 THEN aspect_start = 0
                aspect_end = x_l_len[i] + x_c_len[i]
                if idx < aspect_start:  # left locationed memory
                    if position == 'linear':
                        l = aspect_start.item() - idx
                        memory_l[i][idx] *= (1 - float(l) / int(memory_len[i]))
                    elif position == 'dependency':
                        memory_l[i][idx] *= (1 - float(dep_offset[i][idx]) / int(memory_len[i]))
                elif idx > aspect_end:  # right locationed memory
                    if position == 'linear':
                        l = idx - aspect_end.item()
                        memory_r[i][idx] *= (1 - float(l) / int(memory_len[i]))
                    elif position == 'dependency':
                        memory_r[i][idx] *= (1 - float(dep_offset[i][idx]) / int(memory_len[i]))

        return memory_l, memory_r

    def forward(self, inputs):
        # raw indices for left, center, and right parts
        x_l, x_c, x_r = inputs[0], inputs[1], inputs[2]# , dep_offset, inputs[3]
        x_l_len = torch.sum(x_l != 0, dim=-1)
        x_c_len = torch.sum(x_c != 0, dim=-1)
        x_r_len = torch.sum(x_r != 0, dim=-1)

        # embedding layer
        x_l, x_c, x_r = self.encoder(x_l), self.encoder(x_c), self.encoder(x_r)

        # Memory module:
        # ----------------------------
        # left memory
        if torch.max(x_l_len) == 0:
            memory_l = torch.zeros((x_l.size(0), 1, x_l.size(2))).to(self.args.device)
        else:
            memory_l = torch.zeros((x_l.size(0), torch.max(x_l_len), x_l.size(2))).to(self.args.device) #x_l[:, :torch.max(x_l_len), :]
            memory_l[x_l_len > 0], (_, _) = self.blstm_l(x_l[x_l_len>0], x_l_len[x_l_len>0])
        # print(x_l[x_l_len>0].size(),x_l_len[x_l_len>0].size(),memory_l_tmp.size(), memory_l[x_l_len>0].size())
        # center memory
        # memory_c = x_c[:, :torch.max(x_c_len), :]
        memory_c, (_, _) = self.blstm_c(x_c[x_c_len>0], x_c_len[x_c_len>0])
        # print(memory_c.size(),memory_c_tmp.size())
        # right memory
        # if x_r_len == 0: memory_r = x_r
        # if x_r_len > 0:
        # print((x_r.size(0), torch.max(x_r_len), x_r.size(2)))
        if torch.max(x_r_len) == 0:
            memory_r = torch.zeros((x_r.size(0), 1, x_r.size(2))).to(self.args.device)
        else:
            memory_r = torch.zeros((x_r.size(0), torch.max(x_r_len), x_r.size(2))).to(self.args.device) #x_r[:, :torch.max(x_r_len), :]
            memory_r[x_r_len>0], (_, _) = self.blstm_r(x_r[x_r_len>0], x_r_len[x_r_len>0])

        # Target-Aware memory

        # locationed-memory
        if self.memory_weighter == 'position':
            memory_l, memory_r = self.locationed_memory(memory_l, memory_r, x_l_len,
                                                        x_c_len, x_r_len) #, dep_offset
        # context-attended-memory
        if self.memory_weighter == 'cam':
            pass
        # ----------------------------

        # Aspect vector representation
        x_c_len = torch.tensor(x_c_len, dtype=torch.float).to(self.args.device)
        v_c = torch.sum(memory_c, dim=1)
        v_c = torch.div(v_c, x_c_len.view(x_c_len.size(0), 1))

        # Rotatory attention:
        # ----------------------------
        # [1] Target2Context Attention
        v_l = self.t2c_l_attention(memory_l, v_c).squeeze(dim=1)  # left vector representation
        v_r = self.t2c_r_attention(memory_r, v_c).squeeze(dim=1)  # Right vector representation

        # [2] Context2Target Attention
        v_c_l = self.c2t_l_attention(memory_c, v_l).squeeze(dim=1)  # Left-aware target
        v_c_r = self.c2t_r_attention(memory_c, v_r).squeeze(dim=1)  # Right-aware target
        # ----------------------------

        # sentence representation
        v_s = torch.cat((v_l, v_c_l, v_c_r, v_r), dim=-1)  # dim : (1, 800)
        # v_s = torch.cat((v_l, v_c_l, v_c_r, v_r), dim = 0)     # dim : (4, 300)

        # Classifier
        v_s = self.dropout(v_s)
        out = self.dense(v_s)
        if self.args.softmax:
            out = self.softmax(out)
        return out
