import gzip
import codecs
import re
import numpy as np
from nltk import word_tokenize, sent_tokenize
import sys
import json

reload(sys)
sys.setdefaultencoding('utf8')
num_regex = re.compile('^[+-]?[0-9]+\.?[0-9]*$')
letter_regex = re.compile('^[a-z\']*$')

base_path = sys.path[0] + "/../data/data_doc/"
base_path_aspect = sys.path[0] + "/../data/"


def parse(path):
    g = open(path, 'r')
    gen = []
    line = g.readline()
    while line != "":
        # print(eval(l))
        gen.append(eval(line))
        line = g.readline()
    g.close()
    return gen


def preprocess_review(raw_review):
    raw_review = " ".join(re.findall(r"[\w']+|[.,!?;]", raw_review))
    raw_review = raw_review.replace("-", " ").replace('"', ' ').replace("&quot;", " ").replace("#", " ").replace("/",
                                                                                                                 " ")
    # print(type(raw_review))
    # print(word_tokenize(raw_review), sent_tokenize(raw_review.encode(encoding='ISO-8859-1'), 'english'))
    sents = sent_tokenize(raw_review.lower())
    ans = []
    for sent in sents:
        review_text = ' '.join(word_tokenize(sent.lower()))
        review_text = re.sub(r'((?::|;|=)(?:-)?(?:\)|d|p))', " SMILE", review_text)
        review_text = re.sub(r'\'', '', review_text)
        words = []
        for w in review_text.split():
            if len(w) > 0:
                words.append(w)
                # elif bool(num_regex.match(w)):
                #     words.append(w)
                # elif bool(letter_regex.match(w)):
                #     words.append(w)
        ans.append(' '.join(words))
    return ' BREAKEDU '.join(ans)


def load_aspect(dataset):
    fname = {
        'twitter': {
            'train': base_path_aspect + 'data_processed/Twitter/twitter-train.json',
            'test': base_path_aspect + 'data_processed/Twitter/twitter-test.json'
        },
        'restaurants14': {
            'train': base_path_aspect + 'data_processed/SemEval2014/restaurants-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2014/restaurants-test.json'
        },
        'laptop14': {
            'train': base_path_aspect + 'data_processed/SemEval2014/laptop-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2014/laptop-test.json'
        },
        'restaurants15': {
            'train': base_path_aspect + 'data_processed/SemEval2015/restaurants-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2015/restaurants-test.json'
        },
        'restaurants16': {
            'train': base_path_aspect + 'data_processed/SemEval2016/restaurants-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2016/restaurants-test.json'
        }
    }
    file_train = fname[dataset]["train"]
    file_test = fname[dataset]["test"]
    train_data = json.load(open(file_train, 'r'))
    aspects = []
    for instance in train_data:
        text_instance = instance['text']
        aspect_terms = instance['opinions']['aspect_term']
        for a in aspect_terms:
            aspect = preprocess_review(a['term']).strip()
            if len(aspect) == 0:
                continue
            aspects.append(aspect)

    test_data = json.load(open(file_test, 'r'))
    for instance in test_data:
        text_instance = instance['text']
        aspect_terms = instance['opinions']['aspect_term']
        for a in aspect_terms:
            aspect = preprocess_review(a['term']).strip()
            if len(aspect) == 0:
                continue
            aspects.append(aspect)
    return aspects


def random_extract_balenced_data(gen, dataset, balenced=2000, maxlen_limit=1000):
    out1 = codecs.open(base_path + dataset + '_large/text_{}.txt'.format(balenced * 3), 'w', 'utf-8')
    out2 = codecs.open(base_path + dataset + '_large/label_{}.txt'.format(balenced * 3), 'w', 'utf-8')

    pos, neg, neu = 0, 0, 0
    maxlen, count = 0, 0

    for review in gen:
        # print(review)
        ## for amazon domain
        if dataset == "electronics":
            text = review["reviewText"]
            score = review['overall']

        ## for yelp data
        if dataset == 'yelp':
            text = review["text"]
            score = review['stars']
        # print(text, score)
        # preprocessed = preprocess_review(text)
        try:
            preprocessed = preprocess_review(text)
        except:
            # print("_________", text)
            continue
        # print preprocessed

        tokens = preprocessed.split()
        if len(tokens) > maxlen_limit or len(tokens) < 5:
            continue

        # print(neg, pos, neu)
        if maxlen < len(tokens):
            maxlen = len(tokens)

        if score < 3:
            if neg < balenced:
                count += 1
                out1.write(str(count) + '\t' + preprocessed + '\n')
                out2.write(str(score) + '\n')
                neg += 1

        elif score > 3:
            if pos < balenced:
                count += 1
                out1.write(str(count) + '\t' + preprocessed + '\n')
                out2.write(str(score) + '\n')
                pos += 1

        else:
            if neu < balenced:
                count += 1
                out1.write(str(count) + '\t' + preprocessed + '\n')
                out2.write(str(score) + '\n')
                neu += 1

        if pos >= balenced and neg >= balenced and neu >= balenced:
            break

    return pos, neg, neu, maxlen


def extract_balenced_data_with_aspect(gen, dataset, balenced=2000, maxlen_limit=1000, dataset_aspect="", aspects=[]):
    out1 = codecs.open(base_path + dataset + '_large/text_aspect_{}_{}.txt'.format(balenced * 3, dataset_aspect), 'w',
                       'utf-8')
    out2 = codecs.open(base_path + dataset + '_large/label_aspect_{}_{}.txt'.format(balenced * 3, dataset_aspect), 'w',
                       'utf-8')

    pos, neg, neu = 0, 0, 0
    maxlen, count = 0, 0

    for review in gen:
        ## for amazon domain
        if dataset == "electronics":
            text = review["reviewText"]
            score = review['overall']

        ## for yelp data
        if dataset == 'yelp':
            text = review["text"]
            score = review['stars']

        for aspect in aspects:
            if text.find(aspect) > -1:
                aspect_flag = 1
                break

        if aspect_flag == 0:
            continue

        try:
            preprocessed = preprocess_review(text)
        except:
            continue

        tokens = preprocessed.split()
        if len(tokens) > maxlen_limit or len(tokens) < 5:
            continue

        # print(neg, pos, neu)
        if maxlen < len(tokens):
            maxlen = len(tokens)
        aspect_flag = 0

        if score < 3:
            if neg < balenced:
                count += 1
                out1.write(str(count) + '\t' + preprocessed + '\n')
                out2.write(str(score) + '\n')
                neg += 1

        elif score > 3:
            if pos < balenced:
                count += 1
                out1.write(str(count) + '\t' + preprocessed + '\n')
                out2.write(str(score) + '\n')
                pos += 1

        else:
            if neu < balenced:
                count += 1
                out1.write(str(count) + '\t' + preprocessed + '\n')
                out2.write(str(score) + '\n')
                neu += 1

        if pos >= balenced and neg >= balenced and neu >= balenced:
            break

    return pos, neg, neu, maxlen


def gen_XML(gen, dataset):
    fr_to = open(base_path + dataset + '_large/' + dataset + '.xml', 'w')
    for review in gen:
        ## for amazon domain
        if dataset == "electronics":
            text = review["reviewText"]
            score = review['overall']
            id = review['reviewerID']

        ## for yelp data
        if dataset == 'yelp':
            text = review["text"]
            score = review['stars']
            id = review['review_id']

        fr_to.write('<DOC>\n')
        fr_to.write('<DOCNO>{}</DOCNO>\n{}\n'.format(id, text))
        fr_to.write('</DOC>\n')
    fr_to.close()


def change_id(dataset):
    fr_to = open(base_path + dataset + '_large/' + dataset + '.xml', 'r')
    labels = {}
    num = 0
    for line in fr_to.readlines():
        if line.startswith("<DOCNO>"):
            id = line.strip().replace('<DOCNO>', "").replace('</DOCNO>', "")
            labels[id] = num
            num += 1



def gen_query_XML(dataset):
    fname = {
        'twitter': {
            'train': base_path_aspect + 'data_processed/Twitter/twitter-train.json',
            'test': base_path_aspect + 'data_processed/Twitter/twitter-test.json',
            'query_xml': base_path_aspect + 'data_processed/Twitter/twitter_query.xml'
        },
        'restaurants14': {
            'train': base_path_aspect + 'data_processed/SemEval2014/restaurants-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2014/restaurants-test.json',
            'query_xml': base_path_aspect + 'data_processed/SemEval2014/restaurants14_query.xml'
        },
        'laptop14': {
            'train': base_path_aspect + 'data_processed/SemEval2014/laptop-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2014/laptop-test.json',
            'query_xml': base_path_aspect + 'data_processed/SemEval2014/laptop14_query.xml'
        },
        'restaurants15': {
            'train': base_path_aspect + 'data_processed/SemEval2015/restaurants-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2015/restaurants-test.json',
            'query_xml': base_path_aspect + 'data_processed/SemEval2015/restaurants15_query.xml'
        },
        'restaurants16': {
            'train': base_path_aspect + 'data_processed/SemEval2016/restaurants-train.json',
            'test': base_path_aspect + 'data_processed/SemEval2016/restaurants-test.json',
            'query_xml': base_path_aspect + 'data_processed/SemEval2016/restaurants16_query.xml'
        }
    }
    file_train = fname[dataset]["train"]
    file_test = fname[dataset]["test"]
    fr_to = open(fname[dataset]['query_xml'], 'w')
    train_data = json.load(open(file_train, 'r'))
    num = 0
    for instance in train_data:
        text_instance = instance['text']
        fr_to.write('<top>\n')
        fr_to.write('<num>{}</num>\n'.format(num))
        fr_to.write('<title>{}</title>\n'.format(text_instance))
        fr_to.write('</top>\n')
        num += 1

    test_data = json.load(open(file_test, 'r'))
    for instance in test_data:
        text_instance = instance['text']
        fr_to.write('<top>\n')
        fr_to.write('<num>{}</num>\n'.format(num))
        fr_to.write('<title>{}</title>\n'.format(text_instance))
        fr_to.write('</top>\n')
        num += 1
    fr_to.close()


if __name__ == "__main__":
    gen_query_XML(dataset='restaurants14')
    gen_query_XML(dataset='restaurants15')
    gen_query_XML(dataset='restaurants16')
    gen_query_XML(dataset='laptop14')
    # preprocess_review("We got this GPS for my husband who is an (OTR) over the road trucker.")
    # dataset = 'electronics'
    # file_path = base_path + dataset + '_large/Electronics_5.json'
    # gen = parse(file_path)
    # gen_XML(gen, dataset)
    # pos, neg, neu, maxlen = random_extract_balenced_data(gen, dataset, balenced=10000, maxlen_limit=1000)
    # print(pos, neg, neu)
    #
    # aspects = load_aspect(dataset='laptop14')
    # # print(aspects)
    # pos, neg, neu, maxlen = extract_balenced_data_with_aspect(gen, dataset, balenced=10000, maxlen_limit=1000, aspects=aspects)
    # print(pos, neg, neu)

    # -------------------------------------------------------------
    # dataset = 'yelp'
    # file_path = base_path + dataset + '_large/yelp_academic_dataset_review.json'
    # gen = parse(file_path)
    # gen_XML(gen, dataset)
    # pos, neg, neu, maxlen = random_extract_balenced_data(gen, dataset, balenced=10000, maxlen_limit=1000)
    # print(pos, neg, neu)
    #
    # aspects = load_aspect(dataset='restaurants14')
    # pos, neg, neu, maxlen = extract_balenced_data_with_aspect(gen, dataset, balenced=10000, maxlen_limit=1000,
    #                                                           dataset_aspect="restaurants14", aspects=aspects)
    # print(pos, neg, neu)
    #
    # aspects = load_aspect(dataset='restaurants15')
    # pos, neg, neu, maxlen = extract_balenced_data_with_aspect(gen, dataset, balenced=10000, maxlen_limit=1000,
    #                                                           dataset_aspect="restaurants15", aspects=aspects)
    # print(pos, neg, neu)
    #
    # aspects = load_aspect(dataset='restaurants16')
    # pos, neg, neu, maxlen = extract_balenced_data_with_aspect(gen, dataset, balenced=10000, maxlen_limit=1000,
    #                                                           dataset_aspect="restaurants16", aspects=aspects)
    # print(pos, neg, neu)

    # java -jar segment.jar data_doc/yelp_large/text_30000.txt data_doc/yelp_large/text_30000_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_60000.txt data_doc/yelp_large/text_60000_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_aspect_30000_restaurants14.txt data_doc/yelp_large/text_aspect_30000_restaurants14_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_aspect_60000_restaurants14.txt data_doc/yelp_large/text_aspect_60000_restaurants14_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_aspect_30000_restaurants15.txt data_doc/yelp_large/text_aspect_30000_restaurants15_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_aspect_60000_restaurants15.txt data_doc/yelp_large/text_aspect_60000_restaurants15_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_aspect_30000_restaurants16.txt data_doc/yelp_large/text_aspect_30000_restaurants16_segment.txt
    # java -jar segment.jar data_doc/yelp_large/text_aspect_60000_restaurants16.txt data_doc/yelp_large/text_aspect_60000_restaurants16_segment.txt
