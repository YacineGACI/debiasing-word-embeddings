import torch
import torch.nn as nn


# class NeutralityClassifier(nn.Module):
#     def __init__(self, embedding_dim, hidden_dim, dropout=0.1):
#         super(NeutralityClassifier, self).__init__()
#         self.linear1 = nn.Linear(embedding_dim, hidden_dim)
#         self.linear2 = nn.Linear(hidden_dim, 1)
#         self.relu = nn.ReLu()
#         self.sigmoid = nn.Sigmoid()
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, x):
#         hidden = self.dropout(self.relu(self.linear1(x)))
#         return self.sigmoid(self.linear2(hidden))


class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, encoder_dim, dropout=0.1):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(embedding_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, encoder_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.dropout(dropout)


    def forward(self, x):
        hidden = self.activation(self.linear1(self.dropout(x)))
        return self.activation(self.linear2(self.dropout(hidden)))







class Decoder(nn.Module):
    def __init__(self, decoder_dim, hidden_dim, embedding_dim):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(decoder_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, embedding_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        hidden = self.activation(self.linear1(x))
        return self.linear2(hidden) # I use a linear activation function (identity) because the decoder
                                    # needs to reconstruct the original space

