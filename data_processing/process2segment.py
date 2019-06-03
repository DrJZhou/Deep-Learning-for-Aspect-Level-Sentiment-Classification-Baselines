import os
import numpy as np
import sys
import json
import re
# from Discourse_Segmenter import do_segment
from nltk import word_tokenize
import copy

reload(sys)
sys.setdefaultencoding('UTF8')

base_path = sys.path[0] + "/../data/"
print(base_path)
sentiment_map = {
    'positive': 2,
    'neutral': 1,
    'negative': 0
}

num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
letter_regex = re.compile('^[a-z\']*$')


# num_letter_regex
def preprocess_review(raw_review):
    raw_review = " ".join(re.findall(r"[\w']+|[.,!?;]", raw_review))
    # raw_review = " ".join(re.findall(r'(?:\w+|\W)', raw_review))
    raw_review = raw_review.replace("-", " ").replace('"', ' ').replace("&quot;", " ").replace("#", " ").replace("/",
                                                                                                                 " ")
    # print("1", raw_review)
    review_text = ' '.join(word_tokenize(raw_review.lower()))
    # print("2", review_text)
    # Replace smile emojis with SMILE
    review_text = re.sub(r'((?::|;|=)(?:-)?(?:\)|d|p))', " SMILE", review_text)
    # review_text = re.sub(r' n\'t ', 'n\'t ', review_text)
    # review_text = re.sub(r' \' t ', '\'t ', review_text)
    review_text = re.sub(r'\'', '', review_text)
    # Only keep letters, numbers, ', !, and SMILE
    # print("3", review_text)
    words = []
    for w in review_text.split():
        if len(w.strip()) > 0:
            words.append(w)
            # if w in {'!', 'SMILE', ',', '.'}:
            #     words.append(w)
            # elif bool(num_regex.match(w)):
            #     words.append(w)
            # elif bool(letter_regex.match(w)):
            #     words.append(w)
    return ' '.join(words)


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


class ABSADatesetReader:
    @staticmethod
    def transforer(dataset, fname, fname_to, data_mark, pre_flag=False):
        with open(fname, 'r') as f:
            data = json.load(f)
        if not pre_flag:
            data_segment = []
            base_path_tmp = base_path + "tmp/" + dataset + "_" + data_mark + "_segment.txt"
            tmp_file = open(base_path_tmp, 'rb')
            sentence = ""
            num = 0
            instance = {}
            for line in tmp_file.readlines():
                num += 1
                line = line.replace("z\x00\x00\x04\x00", "").replace("z\x00\x00\x039", "").replace("z\x00\x00\x03\xa6",
                                                                                                   "").replace(
                    "z\x00\x00\x02\xef\xbf\xbd", "").replace("z\x00\x00\x02\xd8", "").replace("z\x00\x00\x01&",
                                                                                              "").replace("w\xde",
                                                                                                          "").replace(
                    "z\x00\x00\x02H", "").replace("z\x00\x00\x03\xf7", "").replace("z\x00\x00\x03+", "").replace(
                    "z\x00\x00\x02\xef", "").replace("\xac\xed\x00\x05", "").replace("w\x92", "")
                tmp = line.strip().split("\t")
                id = tmp[0]
                text = tmp[1]
                # print(id, text)
                if id.find("#<>#") > -1:
                    if text == "###":
                        continue
                    text_split = text.split("####")
                    right, aspect, left = text_split[0], text_split[1], text_split[2]
                    right = " ".join([x.strip() for x in right.strip().split("BREAKEDU")]).strip()
                    aspect = " ".join([x.strip() for x in aspect.strip().split("BREAKEDU")]).strip()
                    left = " ".join([x.strip() for x in left.strip().split("BREAKEDU")]).strip()
                    if right == "###":
                        right = ""
                    if left == "###":
                        left = ""
                    if aspect == "###":
                        aspect = "null"

                    sentence_tmp = " ".join([" ".join(segment) for segment in sentence])

                    from_index = len(right.split(" "))
                    to_index = len(right.split(" ")) + len(aspect.split(" "))
                    if right == "":
                        from_index -= 1
                        to_index -= 1
                    if aspect == "null":
                        from_index = 0
                        to_index = 0
                    length = 0
                    sentence_final = []
                    for segment in sentence:
                        if length > from_index and length < to_index:
                            # print sentence_final, aspect
                            sentence_final[-1] += copy.deepcopy(segment)
                            # print sentence_final, segment
                        else:
                            sentence_final.append(copy.deepcopy(segment))
                        length += len(segment)
                    aspect_instance = {
                        "term": aspect,
                        'from': from_index,
                        'to': to_index,
                        'segments': sentence_final
                    }
                    instance[id] = aspect_instance
                    if np.sum([len(x) for x in sentence_final]) != np.sum([len(x) for x in sentence]):
                        print(sentence_final)
                        print(sentence)
                    if " ".join(sentence_tmp.split(" ")[from_index: to_index]) != aspect and aspect != "null":
                        print aspect, sentence_tmp.split(" ")[from_index: to_index]
                        print(sentence_tmp)
                        print(right, aspect)

                else:
                    sentence = [x.strip().split(" ") for x in text.split("BREAKEDU")]
                    # print(id, num)
                    text_instance = {
                        "text": sentence,
                        "id": id
                    }
                    instance[id] = text_instance

            segment_instances = []
            for inst in data:
                id = str(inst['id'])
                opinion = inst['opinions']
                aspect_terms = opinion['aspect_term']
                num = 0
                segment_aspects = []
                segment_category = opinion["aspect_category"]
                for a in aspect_terms:
                    num += 1
                    sub_id = "{}#<>#{}".format(id, num)
                    if a.has_key("category"):
                        segment_aspects.append({
                            "term": instance[sub_id]['term'],
                            'from': instance[sub_id]['from'],
                            'to': instance[sub_id]['to'],
                            "polarity": a["polarity"],
                            "segments": instance[sub_id]['segments'],
                            'category': a["category"]
                        })
                    else:
                        segment_aspects.append({
                            "term": instance[sub_id]['term'],
                            'from': instance[sub_id]['from'],
                            'to': instance[sub_id]['to'],
                            "segments": instance[sub_id]['segments'],
                            "polarity": a["polarity"]
                        })
                segment_opinion = {"aspect_term": segment_aspects, "aspect_category": segment_category}
                segment_instances.append({
                    "id": id,
                    "text": instance[id]['text'],
                    "opinions": segment_opinion
                })
            json.dump(segment_instances, open(fname_to, 'w'))
        else:
            base_path_tmp = base_path + "tmp/" + dataset + "_" + data_mark + ".txt"
            tmp_file = open(base_path_tmp, 'w')
            for instance in data:
                id = instance['id']
                text_instance = instance['text']  # .replace("&quot;", " ").replace("&gt;", " ").replace("&amp;", " ")
                if dataset == 'twitter':
                    text_instance = text_instance.encode("utf-8")
                tmp_file.write("{}\t{}\n".format(id, preprocess_review(copy.deepcopy(text_instance))))
                opinion = instance['opinions']
                aspect_terms = opinion['aspect_term']
                num = 0
                for a in aspect_terms:
                    num += 1
                    aspect_term = preprocess_review(a['term'])
                    from_index = int(a['from'])
                    to_index = int(a['to'])
                    if aspect_term.lower() == "null":
                        from_index = 0
                        to_index = 0
                    if from_index == 0:
                        right_tmp = "###"
                    else:
                        right_tmp = text_instance[:from_index]
                    if to_index == len(text_instance):
                        left_tmp = "###"
                    else:
                        left_tmp = text_instance[to_index:]
                    # print("1", aspect_term, text_instance[from_index: to_index], left_tmp, right_tmp)
                    if to_index == 0:
                        aspect_tmp = "###"
                    else:
                        aspect_tmp = text_instance[from_index: to_index]
                    # print(text_instance)
                    text_instance_tmp = preprocess_review(copy.deepcopy(text_instance))
                    # print(text_instance_tmp)
                    right = preprocess_review(copy.deepcopy(right_tmp))
                    aspect = preprocess_review(copy.deepcopy(aspect_tmp))
                    left = preprocess_review(copy.deepcopy(left_tmp))
                    if text_instance_tmp != (right + " " + aspect + " " + left).strip():
                        print("1", aspect_term, text_instance[from_index: to_index], left_tmp, right_tmp)
                        print("2", text_instance,)
                        print("3", text_instance_tmp)
                        print("4", (right + " " + aspect + " " + left).strip())
                        print("5", right, aspect, left)
                        print("6", right_tmp, aspect_tmp, left_tmp)
                    if right == "":
                        right = "###"
                    if left == "":
                        left = "###"
                    if aspect == "":
                        aspect = "###"
                    tmp_file.write("{}#<>#{}\t{} #### {} #### {}\n".format(id, num, right, aspect, left))
            tmp_file.close()

    def __init__(self, dataset='twitter'):
        print("preparing {0} dataset...".format(dataset))
        fname = {
            'twitter': {
                'train': base_path + 'data_processed/Twitter/twitter-train.json',
                'test': base_path + 'data_processed/Twitter/twitter-test.json',
                'train_segment': base_path + 'data_processed/Twitter/twitter-train-segment.json',
                'test_segment': base_path + 'data_processed/Twitter/twitter-test-segment.json'
            },
            'restaurants14': {
                'train': base_path + 'data_processed/SemEval2014/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2014/restaurants-test.json',
                'train_segment': base_path + 'data_processed/SemEval2014/restaurants-train-segment.json',
                'test_segment': base_path + 'data_processed/SemEval2014/restaurants-test-segment.json'
            },
            'laptop14': {
                'train': base_path + 'data_processed/SemEval2014/laptop-train.json',
                'test': base_path + 'data_processed/SemEval2014/laptop-test.json',
                'train_segment': base_path + 'data_processed/SemEval2014/laptop-train-segment.json',
                'test_segment': base_path + 'data_processed/SemEval2014/laptop-test-segment.json'
            },
            'restaurants15': {
                'train': base_path + 'data_processed/SemEval2015/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2015/restaurants-test.json',
                'train_segment': base_path + 'data_processed/SemEval2015/restaurants-train-segment.json',
                'test_segment': base_path + 'data_processed/SemEval2015/restaurants-test-segment.json'
            },
            'restaurants16': {
                'train': base_path + 'data_processed/SemEval2016/restaurants-train.json',
                'test': base_path + 'data_processed/SemEval2016/restaurants-test.json',
                'train_segment': base_path + 'data_processed/SemEval2016/restaurants-train-segment.json',
                'test_segment': base_path + 'data_processed/SemEval2016/restaurants-test-segment.json'
            }
        }
        ABSADatesetReader.transforer(dataset, fname[dataset]['train'], fname[dataset]['train_segment'],
                                     data_mark='train')
        ABSADatesetReader.transforer(dataset, fname[dataset]['test'], fname[dataset]['test_segment'], data_mark='test')


def deal_with_doc(filename, tmp_file, fileto):
    fr = open(filename, 'r')
    fr_to = open(tmp_file, 'w')
    num = 0
    for line in fr.readlines():
        num += 1
        data = line.strip()
        fr_to.write("{}\t{}\n".format(num, data))
    fr_to.close()


if __name__ == '__main__':
    # ABSADatesetReader(dataset="restaurants16", embed_dim=300, max_seq_len=80)
    # yelp_file = base_path + "data_doc/yelp_large/text.txt"
    # yelp_file_to = base_path + "data_doc/yelp_large/text_segment.txt"
    # deal_with_doc(filename=yelp_file, tmp_file="data/yelp.txt", fileto=yelp_file_to)
    #
    # electronics_file = base_path + "data_doc/electronics_large/text.txt"
    # electronics_file_to = base_path + "data_doc/electronics_large/text_segment.txt"
    # deal_with_doc(filename=electronics_file, tmp_file="data/electronics.txt", fileto=electronics_file_to)
    ABSADatesetReader(dataset="laptop14")
    ABSADatesetReader(dataset="twitter")
    ABSADatesetReader(dataset="restaurants14")
    ABSADatesetReader(dataset="restaurants15")
    ABSADatesetReader(dataset="restaurants16")
