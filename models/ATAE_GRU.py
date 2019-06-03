import torch
import torch.nn as nn
from layers.Squeeze_embedding import SqueezeEmbedding
from layers.Dynamic_RNN import DynamicRNN
from layers.Attention import BasicAttention,AttentionAspect


class ATAE_GRU(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(ATAE_GRU, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.gru = DynamicRNN(args.embed_dim, args.hidden_dim, num_layers=1, batch_first=True, dropout=args.dropout,
                              rnn_type="GRU")
        self.attention = BasicAttention(hidden_dim=args.hidden_dim, score_function="aspect")
        self.attention = AttentionAspect(embed_dim_k=args.hidden_dim, embed_dim_q=args.embed_dim,
                                         score_function="mlp")
        self.dense = nn.Linear(args.hidden_dim, args.polarities_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        aspect_indices = inputs[1]

        x = self.encoder(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        output, (h_n, _) = self.gru(x, x_len)

        aspect = self.encoder_aspect(aspect_indices)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        aspect_emb = self.squeeze_embedding(aspect, aspect_len)
        aspect_emb = torch.sum(aspect_emb, dim=1) / aspect_len.view(-1, 1).float()
        aspect_emb = aspect_emb.view(aspect_emb.size(0), -1)

        output, _ = self.attention(output, aspect_emb)
        output = output.view(output.size(0), output.size(2))
        output = self.dropout(output)
        output = self.dense(output)
        if self.args.softmax:
            output = self.softmax(output)
        return output
