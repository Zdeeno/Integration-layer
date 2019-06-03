import json
import re
import numpy as np
import torch


filename = "../trump.json"
cuda = torch.device('cuda')


def remove_bad_chars(string):
    string = string.replace('_', ' ')
    string = string.replace('\'', '')
    string = string.replace('’', '')
    string = string.replace('#', 'htgsig')
    string = string.replace('@', 'atsig')

    my_str = re.sub(r'\W+', ' ', string)

    my_str = my_str.lower()
    word_list = my_str.split(' ')

    for i in range(len(word_list)):
        if any(c.isdigit() for c in word_list[i]):
            word_list[i] = "num"

    while '' in word_list:
        word_list.remove('')

    return word_list


def load_dataset():
    with open(filename) as json_file:
        my_json = json.load(json_file)

    dataset = []
    for i in range(len(my_json)):
        dataset.append(remove_bad_chars(my_json[i]['text']))

    return dataset


def create_corpus(dataset):
    corpus = []
    for tweet in dataset:
        for word in tweet:
            if word not in corpus:
                corpus.append(word)
    return corpus


def create_learning_tensors(dataset, corpus):
    c_len = len(corpus)
    ret = []
    for tweet in dataset:
        vecs = []
        for word in tweet:
            vec = np.zeros(c_len)
            vec[corpus.index(word)] = 1
            vecs.append(torch.from_numpy(vec).float().view(1, c_len).cuda())
        ret.append(vecs)
    return ret


def get_learning_data():
    dataset = load_dataset()
    print("dataset loaded")
    corpus = create_corpus(dataset)
    print("corpus created with", len(corpus), "words")
    vectors = create_learning_tensors(dataset, corpus)
    print("learning data obtained")
    return vectors, corpus


if __name__ == '__main__':

    get_learning_data()
