from network import Net
import numpy as np
import torch
import preprocess_data
import random

EPOCH_NUM = 100
HIDDEN_LAYER = 2048
OPTIMIZE = 5


def get_idx(output):
    idx = np.argmax(output.cpu().data.numpy())
    return idx


if __name__ == '__main__':

    cuda = torch.device('cuda')
    torch.cuda.empty_cache()

    dataset, corpus = preprocess_data.get_learning_data()
    c_size = len(corpus)
    network = Net(c_size, [HIDDEN_LAYER], c_size, 0.0003)
    network.to(cuda)
    print(network)

    min_err = float('Inf')

    for epoch in range(EPOCH_NUM):
        print("----- Epoch No. " + str(epoch) + " -----")
        epoch_err = 0
        tweet_num = 1
        random.shuffle(dataset)

        for tweet in dataset:
            network.reset_potentials()
            tweet_err = 0
            pred_idx = None
            tweet = tweet.to_dense().to(cuda)
            for vec_idx in range(tweet.size()[0]):

                if pred_idx is not None:
                    next_idx = get_idx(tweet[vec_idx])
                    target = torch.LongTensor([next_idx]).to(cuda)
                    loss = network.loss(net_output, target)
                    loss.backward()

                    if pred_idx != next_idx:
                        tweet_err += 1
                        epoch_err += 1

                net_output = network(tweet[vec_idx].view(1, -1))
                pred_idx = get_idx(net_output)

            if tweet_num % 5 == 0:
                print("optimizing after tweet num: ", tweet_num)
                network.optimizer.step()
                network.optimizer.zero_grad()

            tweet_num += 1
            # print(len(tweet) - tweet_err, "/", len(tweet))
        print("Epoch No.", epoch, "had", epoch_err, "errors")
        torch.save(network.state_dict(), "./my_net_v2_" + str(epoch))
"""
    # Evaluate
    network.reset_potentials()
    letter, _ = get_next_letter(corpus)
    ret_str = ""
    for i in range(len(corpus)):
        c, next_idx = get_prediction(letter)

        # process input
        # print(c)
        ret_str += c
        letter = np.zeros(LETTER_NUM)
        letter[next_idx] = 1
        letter = torch.from_numpy(letter).float().view(1, LETTER_NUM)

        # feed the network with its last prediction
        letter = network(letter)

        for layer in network.layers:
            if type(layer).__name__ == "Integrating":
                print(layer.potential)
                pass

    print(ret_str)
"""
