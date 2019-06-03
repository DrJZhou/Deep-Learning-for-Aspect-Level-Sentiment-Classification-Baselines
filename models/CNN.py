import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Squeeze_embedding import SqueezeEmbedding


class CNN(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(CNN, self).__init__()
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.convs1 = [
            nn.Conv2d(in_channels=1, out_channels=self.args.kernel_num, kernel_size=(K, self.args.embed_dim), bias=True)
            for K in self.args.kernel_sizes]
        for conv in self.convs1:
            conv = conv.to(self.args.device)
        in_feat = len(self.args.kernel_sizes) * self.args.kernel_num
        self.dense = nn.Linear(in_feat, self.args.polarities_dim, bias=True)

        if args.batch_normalizations is True:
            self.convs1_bn = nn.BatchNorm2d(num_features=self.args.kernel_num)
            self.fc1 = nn.Linear(in_feat, in_feat / 2, bias=True)
            self.fc1_bn = nn.BatchNorm1d(num_features=in_feat / 2)
            self.fc2 = nn.Linear(in_feat / 2, self.args.polarities_dim)
            self.fc2_bn = nn.BatchNorm1d(num_features=self.args.polarities_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices = inputs[0]
        x = self.encoder(text_raw_indices)
        x_len = torch.sum(text_raw_indices != 0, dim=-1)
        emb = self.squeeze_embedding(x, x_len)
        # emb = emb.transpose(0, 1)
        emb = self.dropout(emb)
        emb = emb.unsqueeze(1)
        if self.args.batch_normalizations is True:
            x = [self.convs1_bn(F.tanh(conv(emb))).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        else:
            x = [F.relu(conv(emb)).squeeze(3) for conv in self.convs1]  # [(N,Co,W), ...]*len(Ks)
            x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N,Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        x = self.dropout(x)  # (N,len(Ks)*Co)
        if self.args.batch_normalizations is True:
            x = self.fc1_bn(self.fc1(x))
            output = self.fc2_bn(self.fc2(F.tanh(x)))
        else:
            output = self.dense(x)
        if self.args.softmax:
            output = self.softmax(output)
        return output
