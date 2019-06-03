import torch
import torch.nn as nn
from layers.Dynamic_RNN import DynamicRNN
from layers.Attention import Attention


class IAN(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(IAN, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.lstm_context = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                       dropout=args.dropout, rnn_type="LSTM")
        self.lstm_aspect = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                      dropout=args.dropout, rnn_type="LSTM")
        self.attention_aspect = Attention(args.hidden_dim, score_function='bi_linear')
        self.attention_context = Attention(args.hidden_dim, score_function='bi_linear')
        self.dense = nn.Linear(args.hidden_dim * 2, args.polarities_dim)
        self.droput = nn.Dropout(args.dropout)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        text_raw_len = torch.sum(text_raw_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)

        context = self.encoder(text_raw_indices)
        context = self.droput(context)
        aspect = self.encoder(aspect_indices)
        aspect = self.droput(aspect)
        context, (_, _) = self.lstm_context(context, text_raw_len)
        aspect, (_, _) = self.lstm_aspect(aspect, aspect_len)

        aspect_len = torch.tensor(aspect_len, dtype=torch.float).to(self.args.device)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, aspect_len.view(aspect_len.size(0), 1))

        text_raw_len = torch.tensor(text_raw_len, dtype=torch.float).to(self.args.device)
        context = torch.sum(context, dim=1)
        context = torch.div(context, text_raw_len.view(text_raw_len.size(0), 1))

        aspect_final = self.attention_aspect(aspect, context).squeeze(dim=1)
        context_final = self.attention_context(context, aspect).squeeze(dim=1)

        x = torch.cat((aspect_final, context_final), dim=-1)
        x = self.droput(x)
        out = self.dense(x)
        if self.args.softmax:
            out = self.softmax(out)
        return out
