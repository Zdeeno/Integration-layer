from network import Net
import numpy as np
import torch


CHAR_DIFF = 97
CURR_LETTER = 0
LETTER_NUM = 27
network = Net(LETTER_NUM, [100], LETTER_NUM, 0.0003)
# corpus = "nenaolejuje li julie koleje naolejuji je ja"
# corpus = "mama mele maso"
corpus = "lorem ipsum dolor sit amet consectetuer adipiscing elit maecenas libero curabitur ligula sapien pulvinar a vestibulum quis facilisis vel sapien maecenas sollicitudin"


def get_next_letter(sentence):
    global CURR_LETTER
    if CURR_LETTER < len(sentence):
        letter = sentence[CURR_LETTER]
        CURR_LETTER += 1
        out = np.zeros(LETTER_NUM)
        if letter == ' ':
            idx = 26
            out[LETTER_NUM - 1] = 1
        else:
            idx = ord(letter) - CHAR_DIFF
            out[idx] = 1
        return torch.from_numpy(out).float().view(1, LETTER_NUM), idx
    else:
        CURR_LETTER = 0
        return None, 0


def get_prediction(output):
    idx = np.argmax(output.data.numpy())
    c = chr(idx + CHAR_DIFF)
    if c == '{':
        c = ' '
    return c, idx


if __name__ == '__main__':

    print(network)
    min_err = float('Inf')
    epoch = 1

    while True:
        print("----- Epoch No. " + str(epoch) + " -----")
        network.reset_potentials()
        letter, _ = get_next_letter(corpus)
        err = 0

        while True:
            network.optimizer.zero_grad()
            net_output = network(letter)

            next_letter, next_idx = get_next_letter(corpus)
            if next_letter is None:
                break

            c, ret_idx = get_prediction(net_output)
            # print(c)

            if ret_idx != next_idx:
                err += 1

            loss = network.loss(net_output, torch.LongTensor([next_idx]))
            loss.backward()
            network.optimizer.step()

            letter = next_letter

        epoch += 1
        if err < min_err:
            min_err = err
        print(err, '/', min_err)
        if min_err == 0:
            break

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
