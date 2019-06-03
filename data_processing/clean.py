import re
import string
from bs4 import BeautifulSoup


def strip_punctuation(s):
    # print "output_2", s
    s = s.encode('raw_unicode_escape')
    return s.translate(str.maketrans("", ""), string.punctuation)
    # return ''.join(c for c in s if c not in string.punctuation)


def process_text(x):
    x = x.lower()
    x = x.replace("&quot;", " ")
    x = x.replace('"', " ")
    x = BeautifulSoup(x, "lxml").text
    x = re.sub('[^A-Za-z0-9]+', ' ', x)
    x = x.strip().split(' ')
    # x = [strip_punctuation(y) for y in x]
    ans = []
    for y in x:
        if len(y) == 0:
            continue
        ans.append(y)
    # ptxt = nltk.word_tokenize(ptxt)
    return ans


def clean_str(string, max_seq_len=-1):
    string = string.replace('"', " ")
    string = string.replace("&quot;", " ")
    string = BeautifulSoup(string, "lxml").text
    string = re.sub(r"[^A-Za-z0-9(),!?\"\`]", " ", string)
    string = re.sub(r"\"s", " \"s", string)
    string = re.sub(r"\"ve", " \"ve", string)
    string = re.sub(r"n\"t", " n\"t", string)
    string = re.sub(r"\"re", " \"re", string)
    string = re.sub(r"\"d", " \"d", string)
    string = re.sub(r"\"ll", " \"ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    s = string.strip().lower().split(" ")
    if max_seq_len == -1:
        return s
    elif len(s) > max_seq_len:
        return s[0:max_seq_len]
    else:
        return s


if __name__ == '__main__':
    s = ''
    print(clean_str(s, 205))
    print(process_text(s))
