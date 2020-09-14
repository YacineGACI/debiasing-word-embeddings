import pickle, random
import torch
import torch.nn as nn
from gensim.models import KeyedVectors
from hyperparameters import Hyperparameters as hp
import utils

from models import Encoder, Decoder


torch.manual_seed(0)
random.seed(0)


def train(inputs):
    encoder.train()
    decoder.train()

    total_loss = 0
    total_num = 0

    for input in inputs:

        input = input.to(device)
        
        encoder.zero_grad()
        decoder.zero_grad()

        hidden = encoder(input)
        result = decoder(hidden)

        loss = criterion(input, result)

        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
        total_num += len(input)

    return total_loss / total_num



def test(inputs):
    encoder.eval()
    decoder.eval()

    total_loss = 0
    total_num = 0
    for input in inputs:

        input = input.to(device)

        encoder.zero_grad()
        decoder.zero_grad()

        hidden = encoder(input)
        result = decoder(hidden)

        loss = criterion(input, result)

        total_loss += loss.item()
        total_num += len(input)
    
    return total_loss / total_num





if __name__ == '__main__':

    # Checking the usage of GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() and hp.gpu else "cpu")

    
    # Preparing the data
    print("Loading word embeddings")
    embeddings = KeyedVectors.load_word2vec_format(hp.word_embedding, binary=hp.emb_binary)

    embedding_list = [embeddings[word] for word in embeddings.wv.vocab]
    random.shuffle(embedding_list)

    train_data = torch.split(torch.tensor(embedding_list[hp.pta_dev_num:]), hp.pta_batch_size)
    test_data = torch.split(torch.tensor(embedding_list[:hp.pta_dev_num]), hp.pta_batch_size)


    # Loading the models
    encoder = Encoder(hp.embedding_dim, hp.hidden_dim, hp.latent_dim)
    decoder = Decoder(hp.latent_dim, hp.hidden_dim, hp.embedding_dim)

    encoder.to(device)
    decoder.to(device)

    encoder_optimizer = utils.make_optim(encoder, hp.pta_optimizer, hp.pta_learning_rate, hp.pta_lr_decay, hp.pta_max_grad_norm)
    decoder_optimizer = utils.make_optim(decoder, hp.pta_optimizer, hp.pta_learning_rate, hp.pta_lr_decay, hp.pta_max_grad_norm)

    criterion = nn.MSELoss()

    min_loss = float('inf')

    for epoch in range(1, hp.pta_epochs + 1):
        train_loss = train(train_data)
        eval_loss = test(test_data)

        print("Training {}% --> Train Loss = {}".format((epoch / hp.pta_epochs + 1) * 100, train_loss))
        print("Training {}% --> Test  Loss = {}".format((epoch / hp.pta_epochs + 1) * 100, eval_loss))

        if eval_loss < min_loss:
            min_loss = eval_loss
            min_epoch = epoch

            checkpoint = {
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'hp': hp
                'epoch': epoch
            }
            torch.save(checkpoint, "{}autoencoder_checkpoint".format(hp.save_model))

    

    checkpoint = torch.load('{}autoencoder_checkpoint'.format(hp.save_model))
    torch.save(checkpoint, '{}autoencoder.pt'.format(hp.save_model))

    import os
    os.remove('{}autoencoder_checkpoint'.format(hp.save_model))









