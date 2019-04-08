import torch
import torch.nn as nn
from layers.Dynamic_RNN import DynamicRNN
from layers.Attention import BasicAttention
from layers.Squeeze_embedding import SqueezeEmbedding


def batch_matmul_bias(seq, weight, bias, nonlinearity=''):
    s = None
    bias_dim = bias.size()
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        _s_bias = _s + bias.expand(bias_dim[0], _s.size()[0]).transpose(0, 1)
        if (nonlinearity == 'tanh'):
            _s_bias = torch.tanh(_s_bias)
        _s_bias = _s_bias.unsqueeze(0)
        if (s is None):
            s = _s_bias
        else:
            s = torch.cat((s, _s_bias), 0)
    return s.squeeze()


def batch_matmul(seq, weight, nonlinearity=''):
    s = None
    for i in range(seq.size(0)):
        _s = torch.mm(seq[i], weight)
        if (nonlinearity == 'tanh'):
            _s = torch.tanh(_s)
        _s = _s.unsqueeze(0)
        if (s is None):
            s = _s
        else:
            s = torch.cat((s, _s), 0)
    return s.squeeze()


def attention_mul(rnn_outputs, att_weights):
    attn_vectors = None
    for i in range(rnn_outputs.size(0)):
        h_i = rnn_outputs[i]
        a_i = att_weights[i].unsqueeze(1).expand_as(h_i)
        h_i = a_i * h_i
        h_i = h_i.unsqueeze(0)
        if (attn_vectors is None):
            attn_vectors = h_i
        else:
            attn_vectors = torch.cat((attn_vectors, h_i), 0)
    return torch.sum(attn_vectors, 0)


class HAN(nn.Module):
    def __init__(self, args, embedding_matrix=None, aspect_embedding_matrix=None):
        super(HAN, self).__init__()
        self.position_dim = 100
        self.args = args
        self.encoder = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.encoder_aspect = nn.Embedding.from_pretrained(torch.tensor(aspect_embedding_matrix, dtype=torch.float))
        self.squeeze_embedding = SqueezeEmbedding(batch_first=True)
        self.word_rnn = DynamicRNN(args.embed_dim + self.position_dim, args.hidden_dim, num_layers=1, batch_first=True,
                                   dropout=args.dropout, bidirectional=True, rnn_type="LSTM")
        self.sentence_rnn = DynamicRNN(args.hidden_dim * 2 + self.position_dim, args.hidden_dim, num_layers=1,
                                       batch_first=True,
                                       dropout=args.dropout, bidirectional=True, rnn_type="LSTM")
        self.word_W = nn.Parameter(
            torch.Tensor(args.hidden_dim * 2 + self.position_dim, args.hidden_dim * 2 + self.position_dim))
        self.word_bias = nn.Parameter(torch.Tensor(args.hidden_dim * 2 + self.position_dim, 1))
        self.word_weight_proj = nn.Parameter(torch.Tensor(args.hidden_dim * 2 + self.position_dim, 1))
        self.word_attention = BasicAttention(hidden_dim=args.hidden_dim * 2, score_function="basic")
        self.sentence_W = nn.Parameter(
            torch.Tensor(args.hidden_dim * 2 + self.position_dim, args.hidden_dim * 2 + self.position_dim))
        self.sentence_bias = nn.Parameter(torch.Tensor(args.hidden_dim * 2 + self.position_dim, 1))
        self.sentence_weight_proj = nn.Parameter(torch.Tensor(args.hidden_dim * 2 + self.position_dim, 1))
        self.sentence_attention = BasicAttention(hidden_dim=args.hidden_dim * 2, score_function="basic")
        self.dense = nn.Linear(args.hidden_dim * 4, args.polarities_dim)
        self.word_position_embed = nn.Embedding(1005, self.position_dim)
        self.segment_position_embed = nn.Embedding(25, self.position_dim)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, inputs):
        text_raw_indices, aspect_indices, word_position, segment_position = inputs[0], inputs[1], inputs[2], inputs[3]
        batch_size = text_raw_indices.size(0)
        max_sentence_num = text_raw_indices.size(1)
        document_len = torch.sum(torch.sum(text_raw_indices, dim=-1) != 0, dim=-1)
        text_raw_indices = text_raw_indices.view(-1, text_raw_indices.size(2))
        x_len = torch.sum(text_raw_indices != 0, dim=-1)

        # aspect = self.encoder_aspect(aspect_indices)
        # aspect_len = torch.sum(aspect_indices != 0, dim=-1)
        # aspect_emb = self.squeeze_embedding(aspect, aspect_len)
        # aspect_emb = torch.sum(aspect_emb, dim=1) / aspect_len.view(-1, 1).float()
        # aspect_emb = aspect_emb.view(aspect_emb.size(0), -1)

        word_position = word_position.view(-1, word_position.size(2))
        word_position_emd = self.word_position_embed(word_position)
        x = self.encoder(text_raw_indices)
        x = torch.cat((x, word_position_emd), dim=-1)
        x = self.dropout(x)

        sen_output = torch.zeros((x.size(0), torch.max(x_len), self.args.hidden_dim * 2)).to(self.args.device)
        sen_h_n = torch.zeros((2, x.size(0), self.args.hidden_dim)).to(self.args.device)
        sen_output[x_len > 0], (sen_h_n[:, x_len > 0, :], _) = self.word_rnn(x[x_len > 0], x_len[x_len > 0])

        word_position_emd_tmp = torch.zeros((word_position.size(0), torch.max(x_len), self.position_dim)).to(
            self.args.device)
        word_position_emd_tmp[x_len > 0] = self.squeeze_embedding(word_position_emd[x_len > 0], x_len[x_len > 0])

        # print("sen_output", sen_output)
        # atten_sen_output, _ = self.word_attention(k=sen_output)
        sen_squish = batch_matmul_bias(torch.cat((sen_output, word_position_emd_tmp), dim=-1).transpose(0, 1),
                                       self.word_W, self.word_bias, nonlinearity='tanh')
        sen_attn = batch_matmul(sen_squish, self.word_weight_proj)
        sen_attn_norm = self.softmax(sen_attn.transpose(1, 0))
        atten_sen_output = attention_mul(sen_output.transpose(0, 1), sen_attn_norm.transpose(1, 0))
        # print("atten_sen_output1", atten_sen_output)
        last_sen_output = torch.cat((sen_h_n[0], sen_h_n[1]), dim=1)
        # sen_output = torch.cat((atten_sen_output, last_sen_output), dim=1)
        sen_output = atten_sen_output
        # print("atten_sen_output2", atten_sen_output)
        sen_output = sen_output.view(batch_size, max_sentence_num, -1)
        # sen_output = self.dropout(sen_output)
        segment_position_emb = self.segment_position_embed(segment_position)
        # print("sen_output", sen_output)
        sen_output = torch.cat((sen_output, segment_position_emb), dim=-1)
        doc_output, (doc_h_n, _) = self.sentence_rnn(sen_output, document_len)
        # atten_doc_output, _ = self.sentence_attention(doc_output)
        segment_position_emb_tmp = self.squeeze_embedding(segment_position_emb, document_len)
        doc_squish = batch_matmul_bias(torch.cat((doc_output, segment_position_emb_tmp), dim=-1).transpose(0, 1),
                                       self.sentence_W, self.sentence_bias, nonlinearity='tanh')
        # print("doc_output", doc_output.size())
        # print("doc_squish", doc_squish.size())
        doc_squish = doc_squish.view(doc_output.size(1), doc_output.size(0), -1)
        doc_attn = batch_matmul(doc_squish, self.sentence_weight_proj)
        # print(doc_attn.size())
        doc_attn = doc_attn.view(doc_output.size(1), doc_output.size(0))
        doc_attn_norm = self.softmax(doc_attn.transpose(1, 0))
        atten_doc_output = attention_mul(doc_output.transpose(0, 1), doc_attn_norm.transpose(1, 0))
        last_doc_output = torch.cat((doc_h_n[0], doc_h_n[1]), dim=1)
        doc_output = torch.cat((atten_doc_output, last_doc_output), dim=1)
        # doc_output = atten_doc_output
        output = doc_output
        output = self.dropout(output)
        output = self.dense(output)
        if self.args.softmax:
            output = self.softmax(output)
        return output
