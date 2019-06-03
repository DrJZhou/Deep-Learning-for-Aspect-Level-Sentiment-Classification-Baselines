import torch
import torch.nn as nn
from layers.Dynamic_RNN import DynamicRNN
from layers.Attention import BasicAttention


class AT_GRU(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(AT_GRU, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.gru = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, dropout=args.dropout, rnn_type="GRU")
        self.attention = BasicAttention(hidden_dim=args.hidden_dim)
        self.dense = nn.Linear(args.hidden_dim, args.polarities_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.encoder(text_raw_indices)
        x = self.dropout(x)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        output, (h_n, _) = self.gru(x, x_len)
        output, _ = self.attention(k=output)
        output = self.dropout(output)
        output = self.dense(output)
        if self.args.softmax:
            output = self.softmax(output)
        return output1996
    
