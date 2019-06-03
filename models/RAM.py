from layers.Dynamic_RNN import DynamicRNN
from layers.Attention import Attention
import torch
import torch.nn as nn


class RAM(nn.Module):
    def locationed_memory(self, memory, memory_len):
        # here we just simply calculate the location vector in Model2's manner
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                memory[i][idx] *= (1 - float(idx) / int(memory_len[i]))
        return memory

    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(RAM, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.bi_lstm_context = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                          bidirectional=True, dropout=args.dropout, rnn_type="LSTM")
        self.bi_lstm_aspect = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                         bidirectional=True, dropout=args.dropout, rnn_type="LSTM")
        self.attention = Attention(args.hidden_dim * 2, score_function='mlp')
        self.gru_cell = nn.GRUCell(args.hidden_dim * 2, args.hidden_dim * 2)
        self.dense = nn.Linear(args.hidden_dim * 2, args.polarities_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        memory_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.args.device)

        memory = self.encoder(text_raw_indices)
        memory = self.dropout(memory)
        memory, (_, _) = self.bi_lstm_context(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len)
        aspect = self.encoder(aspect_indices)
        aspect = self.dropout(aspect)
        aspect, (_, _) = self.bi_lstm_aspect(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))

        et = aspect
        for _ in range(self.args.hops):
            it_al = self.attention(memory, et).squeeze(dim=1)
            et = self.gru_cell(it_al, et)
        et = self.dropout(et)
        out = self.dense(et)
        if self.args.softmax:
            out = self.softmax(out)
        return out
