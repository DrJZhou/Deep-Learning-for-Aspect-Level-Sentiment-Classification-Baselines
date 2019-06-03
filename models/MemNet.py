import torch
import torch.nn as nn
from layers.Squeeze_embedding import SqueezeEmbedding
from layers.Attention import Attention


class MemNet(nn.Module):
    def locationed_memory(self, memory, memory_len, left_len, aspect_len):
        # here we just simply calculate the location vector in Model2's manner
        '''
        Updated to calculate location as the absolute diference between context word and aspect
        '''
        for i in range(memory.size(0)):
            for idx in range(memory_len[i]):
                aspect_start = left_len[i] - aspect_len[i]
                if idx < aspect_start:
                    l = aspect_start.item() - idx  # l: absolute distance to the aspect
                else:
                    l = idx + 1 - aspect_start.item()
                memory[i][idx] *= (1 - float(l) / int(memory_len[i]))
        return memory

    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(MemNet, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.attention = Attention(args.embed_dim, score_function='mlp')
        self.x_linear = nn.Linear(args.embed_dim, args.embed_dim)
        self.dense = nn.Linear(args.embed_dim, args.polarities_dim)
        self.dropout = nn.Dropout(args.dropout)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        text_raw_without_aspect_indices, aspect_indices, left_with_aspect_indices = inputs[0], inputs[1], inputs[2]
        left_len = torch.sum(left_with_aspect_indices != 0, dim=-1)
        memory_len = torch.sum(text_raw_without_aspect_indices != 0, dim=-1)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        nonzeros_aspect = torch.tensor(aspect_len, dtype=torch.float).to(self.args.device)

        memory = self.encoder(text_raw_without_aspect_indices)
        memory = self.squeeze_embedding(memory, memory_len)
        # memory = self.locationed_memory(memory, memory_len, left_len, aspect_len)
        aspect = self.encoder(aspect_indices)
        aspect = self.squeeze_embedding(aspect, aspect_len)
        aspect = torch.sum(aspect, dim=1)
        aspect = torch.div(aspect, nonzeros_aspect.view(nonzeros_aspect.size(0), 1))
        x = aspect.unsqueeze(dim=1)
        for _ in range(self.args.hops):
            x = self.x_linear(x)
            out_at = self.attention(memory, x)
            x = out_at + x
        x = x.view(x.size(0), -1)
        # x = self.dropout(x)
        out = self.dense(x)
        if self.args.softmax:
            out = self.softmax(out)
        return out
