import os
import pickle
import numpy as np
from torch.utils.data import Dataset
import sys
import json
from data_processing.clean import clean_str, process_text
import copy

base_path = sys.path[0] + "/data/"
# print(base_path)
sentiment_map = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}


def load_word_vec(path, word2idx=None):
    fin = open(path, 'r', encoding='utf-8', errors='ignore')
    word_vec = {}
    for line in fin.readlines():
        tokens = line.rstrip().split()
        if word2idx is None or tokens[0] in word2idx.keys():
            word_vec[tokens[0]] = np.asarray(tokens[-300:], dtype='float32')
    return word_vec


def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = base_path + 'store/{0}_{1}_embedding_matrix.dat'.format(str(embed_dim), type)
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = base_path + 'store/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else base_path + 'store/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
            else:
                embedding_matrix[i] = np.random.uniform(low=-0.01, high=0.01, size=embed_dim)
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix


def build_aspect_embedding_matrix(word2idx, embed_dim, type):
    aspect_embedding_matrix_file_name = base_path + 'store/{0}_{1}_aspect_embedding_matrix.dat'.format(str(embed_dim),
                                                                                                       type)
    if os.path.exists(aspect_embedding_matrix_file_name):
        print('loading embedding_matrix:', aspect_embedding_matrix_file_name)
        aspect_embedding_matrix = pickle.load(open(aspect_embedding_matrix_file_name, 'rb'))
    else:
        print('loading word vectors...')
        aspect_embedding_matrix = np.zeros((len(word2idx) + 2, embed_dim))  # idx 0 and len(word2idx)+1 are all-zeros
        fname = base_path + 'store/glove.twitter.27B.' + str(embed_dim) + 'd.txt' \
            if embed_dim != 300 else base_path + 'store/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx)
        print('building embedding_matrix:', aspect_embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                aspect_embedding_matrix[i] = vec
            else:
                aspect_embedding_matrix[i] = np.random.uniform(low=-0.01, high=0.01, size=embed_dim)
        pickle.dump(aspect_embedding_matrix, open(aspect_embedding_matrix_file_name, 'wb'))
    return aspect_embedding_matrix


class Tokenizer(object):
    def __init__(self, lower=False, max_seq_len=None, max_aspect_len=None):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.max_aspect_len = max_aspect_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    @staticmethod
    def pad_sequence(sequence, maxlen, dtype='int64', padding='pre', truncating='pre', value=0.):
        x = (np.ones(maxlen) * value).astype(dtype)
        if truncating == 'pre':
            trunc = sequence[-maxlen:]
        else:
            trunc = sequence[:maxlen]
        trunc = np.asarray(trunc, dtype=dtype)
        if padding == 'post':
            x[:len(trunc)] = trunc
        else:
            x[-len(trunc):] = trunc
        return x

    def text_to_sequence(self, text, reverse=False, max_seq_len=-1):
        if self.lower:
            text = text.lower()
        words = text.split()
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        pad_and_trunc = 'post'  # use post padding together with torch.nn.utils.rnn.pack_padded_sequence
        if reverse:
            sequence = sequence[::-1]
        if max_seq_len == -1:
            max_seq_len = self.max_seq_len
        return Tokenizer.pad_sequence(sequence, max_seq_len, dtype='int64', padding=pad_and_trunc,
                                      truncating=pad_and_trunc)


class ABSADataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


class ABSADatesetReader:
    @staticmethod
    def __read_text__(fname, dataset):
        with open(fname, 'r') as f:
            data = json.load(f)
        text = ''
        aspect_text = ''
        max_sentence_len = 0.0
        max_term_len = 0.0
        for instance in data:
            text_instance = instance['text']
            if dataset == "twitter":
                text_instance = text_instance.encode("utf-8")
            # print(text_instance)
            opinion = instance['opinions']
            aspect_terms = opinion['aspect_term']
            for a in aspect_terms:
                aspect = a['term']
                polarity = a['polarity']
                if polarity == "conflict":
                    continue
                from_index = int(a['from'])
                to_index = int(a['to'])
                aspect_clean = " ".join(process_text(aspect))
                if aspect == "null":
                    from_index = 0
                    to_index = 0
                left = text_instance[:from_index]
                right = text_instance[to_index:]
                aspect_tmp = text_instance[from_index: to_index]
                if dataset == "twitter":
                    left = left.decode("utf-8")
                    right = right.decode("utf-8")
                    aspect_tmp = aspect_tmp.decode("utf-8")
                if aspect != aspect_tmp and aspect != 'NULL':
                    print(aspect, text_instance[from_index: to_index])
                left_clean = " ".join(process_text(left))
                right_clean = " ".join(process_text(right))
                text_raw = left_clean + " " + aspect_clean + " " + right_clean
                if len(text_raw.split(" ")) > max_sentence_len:
                    max_sentence_len = len(text_raw.split(" "))
                # print(aspect_clean)
                if len(aspect_clean.split(" ")) > max_term_len:
                    max_term_len = len(aspect_clean.split(" "))
                text += text_raw + " "
                aspect_text += aspect_clean + " "
        return text.strip(), aspect_text.strip(), max_sentence_len, max_term_len

    @staticmethod
    def __read_data__(fname, tokenizer, dataset):
        with open(fname, 'r') as f:
            data = json.load(f)

        all_data = []
        for instance in data:
            text_instance = instance['text']
            if dataset == "twitter":
                text_instance = text_instance.encode("utf-8")
            opinion = instance['opinions']
            aspect_terms = opinion['aspect_term']
            for a in aspect_terms:
                aspect = a['term']
                polarity = a['polarity']
                if polarity == "conflict":
                    continue
                from_index = int(a['from'])
                to_index = int(a['to'])
                aspect = " ".join(process_text(aspect))
                if aspect == "null":
                    from_index = 0
                    to_index = 0

                left = text_instance[:from_index]
                right = text_instance[to_index:]
                if dataset == "twitter":
                    left = left.decode("utf-8")
                    right = right.decode("utf-8")
                text_left = " ".join(process_text(left))
                text_right = " ".join(process_text(right))
                text_raw_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
                text_raw_without_aspect_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
                text_left_indices = tokenizer.text_to_sequence(text_left)
                text_left_with_aspect_indices = tokenizer.text_to_sequence(text_left + " " + aspect)
                text_right_indices = tokenizer.text_to_sequence(text_right, reverse=True)
                text_right_with_aspect_indices = tokenizer.text_to_sequence(" " + aspect + " " + text_right,
                                                                            reverse=True)
                aspect_indices = tokenizer.text_to_sequence(aspect, max_seq_len=tokenizer.max_aspect_len)
                polarity = sentiment_map[polarity]
                data = {
                    'text_raw_indices': text_raw_indices,
                    'text_raw_without_aspect_indices': text_raw_without_aspect_indices,
                    'text_left_indices': text_left_indices,
                    'text_left_with_aspect_indices': text_left_with_aspect_indices,
                    'text_right_indices': text_right_indices,
                    'text_right_with_aspect_indices': text_right_with_aspect_indices,
                    'aspect_indices': aspect_indices,
                    'polarity': polarity,
                }

                all_data.append(data)
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300, max_seq_len=-1):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'twitter': {
                'train': base_path + 'data_processed/Twitter/twitter-train.json',
                'test': base_path + 'data_processed/Twitter/twitter-test.json'
            },
            'restaurants14': {
                'train': base_path + 'data_processed/SemEval2014/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2014/restaurants-test.json'
            },
            'laptop14': {
                'train': base_path + 'data_processed/SemEval2014/laptop-train.json',
                'test': base_path + 'data_processed/SemEval2014/laptop-test.json'
            },
            'restaurants15': {
                'train': base_path + 'data_processed/SemEval2015/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2015/restaurants-test.json'
            },
            'restaurants16': {
                'train': base_path + 'data_processed/SemEval2016/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2016/restaurants-test.json'
            }
        }
        text_train, aspect_text_train, max_seq_len_train, max_term_len_train = ABSADatesetReader.__read_text__(
            fname[dataset]['train'], dataset=dataset)
        text_test, aspect_text_test, max_seq_len_test, max_term_len_test = ABSADatesetReader.__read_text__(
            fname[dataset]['test'], dataset=dataset)
        text = text_train + " " + text_test
        # aspect_text = aspect_text_train + " " + aspect_text_test
        if max_seq_len < 0:
            max_seq_len = max_seq_len_train
        tokenizer_text = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_term_len_train)
        tokenizer_text.fit_on_text(text.lower())
        # tokenizer_aspect = Tokenizer(max_seq_len=max_seq_len, max_aspect_len=max_term_len_train)
        # tokenizer_aspect.fit_on_text(aspect_text.lower())
        # print tokenizer_aspect.word2idx
        self.embedding_matrix = build_embedding_matrix(tokenizer_text.word2idx, embed_dim, dataset)
        self.aspect_embedding_matrix = copy.deepcopy(self.embedding_matrix)
        # #build_aspect_embedding_matrix(tokenizer_text.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer_text, dataset=dataset))
        self.test_data = ABSADataset(
            ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer_text, dataset=dataset))
        self.dev_data = ABSADataset([])


if __name__ == '__main__':
    ABSADatesetReader(dataset="twitter", embed_dim=300, max_seq_len=80)
    ABSADatesetReader(dataset="laptop14", embed_dim=300, max_seq_len=80)
    ABSADatesetReader(dataset="restaurants14", embed_dim=300, max_seq_len=80)
    ABSADatesetReader(dataset="restaurants15", embed_dim=300, max_seq_len=80)
    ABSADatesetReader(dataset="restaurants16", embed_dim=300, max_seq_len=80)
