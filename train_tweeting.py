from network import Net
import numpy as np
import torch
import preprocess_data

EPOCH_NUM = 10
HIDDEN_LAYER = 512


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
        network.reset_potentials()

        for tweet in dataset:
            err = 0
            pred_idx = None
            for vector in tweet:
                vector = vector.to_dense().view(1, c_size)
                vector = vector.to(cuda)
                network.optimizer.zero_grad()

                if pred_idx is not None:
                    next_idx = get_idx(vector)
                    target = torch.LongTensor([next_idx]).to(cuda)
                    loss = network.loss(net_output, target)
                    loss.backward()
                    network.optimizer.step()

                    if pred_idx != next_idx:
                        err += 1

                net_output = network(vector)
                pred_idx = get_idx(net_output)

            print(len(tweet) - err, "/", len(tweet))
        torch.save(network.state_dict(), "./my_net_" + str(epoch))
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
