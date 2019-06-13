from network import Net
import preprocess_data
import torch
import numpy as np

INIT = "nuclear weapons"
cuda = torch.device('cuda')


def get_idx(output):
    idx = np.argmax(output.cpu().data.numpy())
    return idx


def create_vec(idx, corpus_len):
    return torch.sparse.FloatTensor(torch.LongTensor([[idx]]), torch.FloatTensor([1.0]), torch.Size([corpus_len]))


def save_corpus(corpus):
    with open("corpus.txt", "w") as f:
        for s in corpus:
            f.write(s + "\n")


def load_corpus():
    corpus = []
    with open("corpus.txt", "r") as f:
        for line in f:
            corpus.append(line.strip())
    return corpus


if __name__ == '__main__':
    corpus = load_corpus()
    network = Net(len(corpus), [2048], len(corpus), 0.01)
    network.load_state_dict(torch.load("./my_net_v2_5"))
    network.to(cuda)
    final_tweet = ""

    init_list = INIT.split(" ")
    for word in init_list:
        idx = corpus.index(word)
        input = create_vec(idx, len(corpus))
        output = network(input.to_dense().view(1, -1).to(cuda))
        final_tweet += word + " "

    idx = get_idx(output)

    while True:
        input = create_vec(idx, len(corpus))
        output = network(input.to_dense().view(1, -1).to(cuda))
        idx = get_idx(output)
        out_word = corpus[idx]
        final_tweet += out_word + " "
        if out_word == "EOT":
            break

    print(final_tweet)
