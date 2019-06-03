import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Squeeze_embedding import SqueezeEmbedding


class GCAE(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(GCAE, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.convs1 = nn.ModuleList([nn.Conv1d(args.embed_dim, args.kernel_num, K) for K in args.kernel_sizes])
        self.convs2 = nn.ModuleList([nn.Conv1d(args.embed_dim, args.kernel_num, K) for K in args.kernel_sizes])
        self.convs3 = nn.ModuleList([nn.Conv1d(args.embed_dim, args.kernel_num, K, padding=K - 2) for K in [3]])
        self.fc1 = nn.Linear(len(args.kernel_sizes) * args.kernel_num, args.polarities_dim)
        self.fc_aspect = nn.Linear(args.kernel_num, args.kernel_num)
        self.dropout = nn.Dropout(args.dropout)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        text_raw_indices, aspect_indices = inputs[0], inputs[1]
        x = self.encoder(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        emb = self.squeeze_embedding(x, x_len)

        aspect = self.encoder(aspect_indices)
        aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        aspect_emb = self.squeeze_embedding(aspect, aspect_len)

        aa = [F.relu(conv(aspect_emb.transpose(1, 2))) for conv in self.convs3]  # [(N,Co,L), ...]*len(Ks)
        aa = [F.max_pool1d(a, a.size(2)).squeeze(2) for a in aa]
        aspect_v = torch.cat(aa, 1)

        x = [F.tanh(conv(emb.transpose(1, 2))) for conv in self.convs1]  # [(N,Co,L), ...]*len(Ks)
        y = [F.relu(conv(emb.transpose(1, 2)) + self.fc_aspect(aspect_v).unsqueeze(2)) for conv in self.convs2]
        x = [i * j for i, j in zip(x, y)]

        # pooling method
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        # x = [F.adaptive_max_pool1d(i, 2) for i in x]
        # x = [i.view(i.size(0), -1) for i in x]

        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        output = self.fc1(x)  # (N,C)
        if self.args.softmax:
            output = self.softmax(output)
        return output
