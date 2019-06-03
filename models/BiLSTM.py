import torch
import torch.nn as nn
from layers.Dynamic_RNN import DynamicRNN


class BiLSTM(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(BiLSTM, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.lstm = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, dropout=args.dropout,
                               bidirectional=True, rnn_type="LSTM")
        self.dense = nn.Linear(args.hidden_dim * 2, args.polarities_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.dropout(self.encoder(text_raw_indices))
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        _, (h_n, _) = self.lstm(x, x_len)
        output = torch.cat((h_n[0], h_n[1]), dim=1)
        output = self.dropout(output)
        output = self.dense(output)
        if self.args.softmax:
            output = self.softmax(output)
        return output
