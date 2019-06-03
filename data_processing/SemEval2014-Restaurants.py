#!/usr/bin/env python
import sys
reload(sys)
sys.setdefaultencoding("UTF8")
try:
    import xml.etree.ElementTree as ET, getopt, logging, sys, random, re, copy
    from xml.sax.saxutils import escape
except:
    sys.exit('Some package is missing... Perhaps <re>?')
import json
from clean import clean_str

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

base_path_from = sys.path[0] + "/../data/data_orign/SemEval2014-Task4/"
base_path_to = sys.path[0] + "/../data/data_processed/SemEval2014/"

# Stopwords, imported from NLTK (v 2.0.4)
stopwords = set(
    ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
     'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
     'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
     'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
     'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
     'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
     'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
     'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
     'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now'])


def fd(counts):
    '''Given a list of occurrences (e.g., [1,1,1,2]), return a dictionary of frequencies (e.g., {1:3, 2:1}.)'''
    d = {}
    for i in counts: d[i] = d[i] + 1 if i in d else 1
    return d


freq_rank = lambda d: sorted(d, key=d.get, reverse=True)
'''Given a map, return ranked the keys based on their values.'''


def fd2(counts):
    '''Given a list of 2-uplets (e.g., [(a,pos), (a,pos), (a,neg), ...]), form a dict of frequencies of specific items (e.g., {a:{pos:2, neg:1}, ...}).'''
    d = {}
    for i in counts:
        # If the first element of the 2-uplet is not in the map, add it.
        if i[0] in d:
            if i[1] in d[i[0]]:
                d[i[0]][i[1]] += 1
            else:
                d[i[0]][i[1]] = 1
        else:
            d[i[0]] = {i[1]: 1}
    return d


def validate(filename):
    '''Validate an XML file, w.r.t. the format given in the 4th task of **SemEval '14**.'''
    elements = ET.parse(filename).getroot().findall('sentence')
    aspects = []
    for e in elements:
        for eterms in e.findall('aspectTerms'):
            if eterms is not None:
                for a in eterms.findall('aspectTerm'):
                    aspects.append(Aspect('', '', []).create(a).term)
    return elements, aspects


fix = lambda text: escape(text.encode('utf8')).replace('\"', '&quot;')
'''Simple fix for writing out text.'''


# Dice coefficient
def dice(t1, t2, stopwords=[]):
    tokenize = lambda t: set([w for w in t.split() if (w not in stopwords)])
    t1, t2 = tokenize(t1), tokenize(t2)
    return 2. * len(t1.intersection(t2)) / (len(t1) + len(t2))


class Category:
    '''Category objects contain the term and polarity (i.e., pos, neg, neu, conflict) of the category (e.g., food, price, etc.) of a sentence.'''

    def __init__(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity

    def create(self, element):
        self.term = element.attrib['category']
        self.polarity = element.attrib['polarity']
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Aspect:
    '''Aspect objects contain the term (e.g., battery life) and polarity (i.e., pos, neg, neu, conflict) of an aspect.'''

    def __init__(self, term, polarity, offsets):
        self.term = term
        self.polarity = polarity
        self.offsets = offsets

    def create(self, element):
        self.term = element.attrib['term']
        self.polarity = element.attrib['polarity']
        self.offsets = {'from': str(element.attrib['from']), 'to': str(element.attrib['to'])}
        return self

    def update(self, term='', polarity=''):
        self.term = term
        self.polarity = polarity


class Instance:
    '''An instance is a sentence, modeled out of XML (pre-specified format, based on the 4th task of SemEval 2014).
    It contains the text, the aspect terms, and any aspect categories.'''

    def __init__(self, element):
        self.text = element.find('text').text
        self.id = element.get('id')
        self.aspect_terms = [Aspect('', '', offsets={'from': '', 'to': ''}).create(e) for es in
                             element.findall('aspectTerms') for e in es if
                             es is not None]
        self.aspect_categories = [Category(term='', polarity='').create(e) for es in element.findall('aspectCategories')
                                  for e in es if
                                  es is not None]

    def get_aspect_terms(self):
        return [a.term.lower() for a in self.aspect_terms]

    def get_aspect_categories(self):
        return [c.term.lower() for c in self.aspect_categories]

    def add_aspect_term(self, term, polarity='', offsets={'from': '', 'to': ''}):
        a = Aspect(term, polarity, offsets)
        self.aspect_terms.append(a)

    def add_aspect_category(self, term, polarity=''):
        c = Category(term, polarity)
        self.aspect_categories.append(c)


class Corpus:
    '''A corpus contains instances, and is useful for training algorithms or splitting to train/test files.'''

    def __init__(self, elements):
        self.corpus = [Instance(e) for e in elements]
        self.size = len(self.corpus)
        self.aspect_terms_fd = fd([a for i in self.corpus for a in i.get_aspect_terms()])
        self.top_aspect_terms = freq_rank(self.aspect_terms_fd)
        self.texts = [t.text for t in self.corpus]

    def echo(self):
        print '%d instances\n%d distinct aspect terms' % (len(self.corpus), len(self.top_aspect_terms))
        print 'Top aspect terms: %s' % (', '.join(self.top_aspect_terms[:10]))

    def clean_tags(self):
        for i in range(len(self.corpus)):
            self.corpus[i].aspect_terms = []

    def split(self, threshold=0.8, shuffle=False):
        '''Split to train/test, based on a threshold. Turn on shuffling for randomizing the elements beforehand.'''
        clone = copy.deepcopy(self.corpus)
        if shuffle: random.shuffle(clone)
        train = clone[:int(threshold * self.size)]
        test = clone[int(threshold * self.size):]
        return train, test

    def write_out(self, filename, instances, short=True):
        with open(filename, 'w') as o:
            o.write('<sentences>\n')
            for i in instances:
                o.write('\t<sentence id="%s">\n' % (i.id))
                o.write('\t\t<text>%s</text>\n' % (i.text))
                o.write('\t\t<aspectTerms>\n')
                if not short:
                    for a in i.aspect_terms:
                        o.write('\t\t\t<aspectTerm term="%s" polarity="%s" from="%s" to="%s"/>\n' % (
                            a.term, a.polarity, a.offsets['from'], a.offsets['to']))
                o.write('\t\t</aspectTerms>\n')
                o.write('\t\t<aspectCategories>\n')
                if not short:
                    for c in i.aspect_categories:
                        o.write('\t\t\t<aspectCategory category="%s" polarity="%s"/>\n' % (fix(c.term), c.polarity))
                o.write('\t\t</aspectCategories>\n')
                o.write('\t</sentence>\n')
            o.write('</sentences>')

    def write_out_json(self, filename, instances):
        fr_to = open(filename, 'w')
        ans = []
        for i in instances:
            aspect_term = []
            aspect_category = []
            for a in i.aspect_terms:
                aspect_term.append({
                    "term": a.term,
                    "polarity": a.polarity,
                    "from": a.offsets["from"],
                    "to": a.offsets["to"]
                })
                # if a.term != i.text[int(a.offsets['from']): int(a.offsets['to'])]:
                #     print(a.term, i.text[int(a.offsets['from']): int(a.offsets['to'])])
            for c in i.aspect_categories:
                aspect_category.append({
                    "category": fix(c.term),
                    "polarity": c.polarity
                })
            opinions = {"aspect_term": aspect_term, "aspect_category": aspect_category}
            ans.append({
                "id": i.id,
                "text": i.text,
                "opinions": opinions
            })
        import json
        json.dump(ans, fr_to)


def standardization():
    # trainfile, testfile= None, None
    trainfile = base_path_from + "train/Restaurants_Train_v2.xml"
    testfile = base_path_from + "test/Restaurants_Test.xml"
    # Examine if the file is in proper XML format for further use.
    print 'Validating the file...'
    try:
        elements, aspects = validate(trainfile)
        print 'PASSED! This train has: %d sentences, %d aspect term occurrences, and %d distinct aspect terms.' % (
            len(elements), len(aspects), len(list(set(aspects))))
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    try:
        elements, aspects = validate(testfile)
        print 'PASSED! This test has: %d sentences, %d aspect term occurrences, and %d distinct aspect terms.' % (
            len(elements), len(aspects), len(list(set(aspects))))
    except:
        print "Unexpected error:", sys.exc_info()[0]
        raise

    # Get the corpus and split into train/test.
    corpus_train = Corpus(ET.parse(trainfile).getroot().findall('sentence'))
    # domain_name = 'laptops'
    corpus_test = Corpus(ET.parse(testfile).getroot().findall('sentence'))
    domain_name = 'restaurants'
    corpus_train.write_out(base_path_to + '%s-train.xml' % domain_name, corpus_train.corpus, short=False)
    corpus_test.write_out(base_path_to + '%s-test.xml' % domain_name, corpus_test.corpus, short=False)

    corpus_train.write_out_json(base_path_to + '%s-train.json' % domain_name, corpus_train.corpus)
    corpus_test.write_out_json(base_path_to + '%s-test.json' % domain_name, corpus_test.corpus)


sentiment_index = {
    'negative': 0,
    'neutral': 1,
    'positive': 2
}
index_sentiment = {
    0: 'negative',
    1: 'neutral',
    2: 'positive'
}


def clear_data(data):
    instance_clean = []
    for instance in data:
        id = instance['id']
        text = instance['text']
        opinion = instance['opinions']
        aspect_terms = opinion['aspect_term']
        text_clean = clean_str(text)
        opinion_clean = []
        for a in aspect_terms:
            aspect = a['term']
            polarity = a['polarity']
            if polarity == "conflict":
                continue
            from_index = int(a['from'])
            to_index = int(a['to'])
            aspect_clean = clean_str(aspect)
            start_clean = clean_str(text[:from_index])
            if to_index == 0:
                opinion_clean.append({'aspect': aspect_clean, 'polarity': polarity, 'from': 0,
                                      'to': 0})
            else:
                opinion_clean.append({'aspect': aspect_clean, 'polarity': polarity, 'from': len(start_clean),
                                      'to': len(start_clean) + len(aspect_clean)})
        if len(opinion_clean) == 0:
            continue
        instance_clean.append({'id': id, 'text': text_clean, 'opinion': opinion_clean})
    return instance_clean


def statistic(data, fr_to, data_type="train"):
    num = len(data)
    aspects = {}
    sentence_length = 0.0
    polarity_num = {}
    term_length = 0.0
    term_num = 0.0
    for instance in data:
        text = instance['text']
        sentence_length += len(text)
        id = instance['id']
        opinions = instance['opinion']
        term_num += len(opinions)
        for a in opinions:
            aspect = a['aspect']
            from_index = a['from']
            to_index = a['to']
            polarity = a['polarity']
            aspects[" ".join(aspect)] = aspects.get(" ".join(aspect), 0) + 1
            polarity_num[polarity] = polarity_num.get(polarity, 0) + 1
    avg_sentence_length = sentence_length / num

    aspects = sorted(aspects.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
    term_num_different = len(aspects)
    for tmp in aspects:
        term_length += len(tmp[0].split(" "))
    avg_term_length = term_length / term_num_different
    avg_term_num = term_num / num
    print term_num, num
    # fr_to.write(data_type+"\n")
    total_NNP = polarity_num['negative'] + polarity_num['neutral'] + polarity_num['positive']
    fr_to.write("{} sample numbers: {}\n".format(data_type, num))
    fr_to.write("{} average target number/sample: {}\n".format(data_type, avg_term_num))
    fr_to.write("{} total target number: {}\n".format(data_type, term_num_different))
    fr_to.write("{} average target length: {}\n".format(data_type, avg_term_length))
    fr_to.write("{} average sample length: {}\n".format(data_type, avg_sentence_length))
    fr_to.write("{} negative,neutral,positive number: {},{},{}; rate: {},{},{}\n".format(data_type,
                                                                                         polarity_num['negative'],
                                                                                         polarity_num['neutral'],
                                                                                         polarity_num['positive'],
                                                                                         polarity_num[
                                                                                             'negative'] * 1.0 / total_NNP,
                                                                                         polarity_num[
                                                                                             'neutral'] * 1.0 / total_NNP,
                                                                                         polarity_num[
                                                                                             'positive'] * 1.0 / total_NNP))
    fr_to.write("{} top 10 aspects: ".format(data_type))
    for aspect in aspects[:10]:
        fr_to.write("{}:{};".format(aspect[0], aspect[1]))
    fr_to.write("\n")
    fr_to.write("----------------------------------------\n")
    return num, avg_term_num, term_num_different, avg_term_length, avg_sentence_length, polarity_num, aspects[:10]


def processing():
    domain_name = 'restaurants'
    train_file = base_path_to + '%s-train.json' % domain_name
    test_file = base_path_to + '%s-test.json' % domain_name
    train_data = json.load(open(train_file, 'r'))
    test_data = json.load(open(test_file, 'r'))
    all_data = train_data + test_data
    train_data_clean = clear_data(train_data)
    test_data_clean = clear_data(test_data)
    all_data_clean = clear_data(all_data)

    statistic_file = base_path_to + '%s-statistic.txt' % domain_name
    statistic_fr_to = open(statistic_file, 'w')
    statistic(train_data_clean, statistic_fr_to, 'train')
    statistic(test_data_clean, statistic_fr_to, 'test')
    statistic(all_data_clean, statistic_fr_to, 'train+test')
    statistic_fr_to.close()


if __name__ == "__main__":
    standardization()
    processing()
