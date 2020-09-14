import torch
from hyperparameters import Hyperparameters as hp

def remove_oov_words(data_filepath, embeddings):
    # Returns the list of words in @data_filepath that belong to the vocabulary of @embeddings
    wordlist = []
    with open(data_filepath, 'r') as f:
        for line in f.readlines():
            for word in line.strip().split('\t'):
                if word in embeddings:
                    wordlist.append(word)
    return wordlist
    


def shuffle_data(l):
    permutation = torch.randperm(len(l))
    return [l[index] for index in permutation]


def create_train_dev(l):
    l = shuffle_data(l)
    train = l[:-hp.dev_num]
    dev = l[-hp.dev_num:]
    return train, dev


def make_optim(model, optimizer, learning_rate, lr_decay, max_grad_norm):
    model_optim = torch.optim.Optim(optimizer, learning_rate, lr_decay, max_grad_norm)
    model_optim.set_parameters(model.parameters())
    return model_optim