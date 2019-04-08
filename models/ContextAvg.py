import torch
import torch.nn as nn
from layers.Squeeze_embedding import SqueezeEmbedding


class ContextAvg(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(ContextAvg, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.dense = nn.Linear(args.embed_dim, args.polarities_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.encoder(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        emb = self.squeeze_embedding(x, x_len)
        output = torch.sum(emb, dim=1) / x_len.view(-1, 1).float()
        output = output.view(output.size(0), -1)
        output = self.dropout(output)
        output = self.dense(output)
        if self.args.softmax:
            output = self.softmax(output)
        return output
