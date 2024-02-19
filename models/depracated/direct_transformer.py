import torch
import torch.nn as nn
import math 
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
import torch.optim as optim
from torch import Tensor
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
import math

from .utils import *




class DirectTransformerModel(nn.Module):
    def __init__(self,
                 encoder_input_dim: int, # dimension of the input to the encoder
                 num_encoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 tgt_vocab_size: int, # number of output classes
                 max_len: int, # maximum length of the encoder input
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation=torch.nn.ReLU()): # activation function of the input dim matching linear layers
        super(DirectTransformerModel, self).__init__()

        # parameters
        self.activation = activation

        # positional encoding
        self.encoder_positional_encoding = PositionalEncoding(
            encoder_input_dim, dropout=dropout, max_len=max_len)
        
        # encoder input transformation to model dimension
        self.encoder_input_transform = torch.nn.Linear(in_features=encoder_input_dim, out_features=emb_size)

        # encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=emb_size, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers, num_layers=num_encoder_layers
        )

        # output layer    
        self.fc_output = nn.Linear(emb_size, tgt_vocab_size)

    def forward(self, src: Tensor):
        
        # src transformation
        src_emb = self.encoder_positional_encoding(src) # add positional encoding to the kinematics data
        src_emb = self.encoder_input_transform(src_emb)
        if self.activation:
            src_emb = self.activation(src_emb)

        # encoder
        encoded = self.transformer_encoder(src_emb)

        # output
        out = self.fc_output(encoded)

        return out