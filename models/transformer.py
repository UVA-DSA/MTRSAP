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


class TransformerEncoderDecoderModel(nn.Module):
    def __init__(self,
                 encoder_input_dim: int, # dimension of the input to the encoder
                 decoder_input_dim: int, # dimension of the input to the decoder (before embedding)
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_dim: int,
                 nhead: int,
                 tgt_vocab_size: int, # number of output classes
                 tgt_reg_size: int, # number of regression outputs
                 max_encoder_len: int, # maximum length of the encoder input
                 max_decoder_len: int, # maximum length of the decoder input
                 decoder_embedding_dim: int = -1, # embedding dimension for the decoder input (if > 0)
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 activation=torch.nn.ReLU()): # activation function of the input dim matching linear layers
        super(TransformerEncoderDecoderModel, self).__init__()

        # parameters
        self.decoder_embedding_dim = decoder_embedding_dim
        self.activation = activation

        # positional encoding
        self.encoder_positional_encoding = PositionalEncoding(
            encoder_input_dim, dropout=dropout, max_len=max_encoder_len)
        
        # encoder input transformation to model dimension
        self.encoder_input_transform = torch.nn.Linear(in_features=encoder_input_dim, out_features=emb_dim)
        
        # edecoder
        dim = decoder_embedding_dim if decoder_embedding_dim > 0 else decoder_input_dim
        self.decoder_positional_encoding = PositionalEncoding(
            dim, dropout=dropout, max_len=max_decoder_len)

        # custom encoder
        self.transformer = Transformer(d_model=emb_dim,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        
        # decoder embeddings and transformation
        if self.decoder_embedding_dim > 0:
            self.tgt_tok_emb = TokenEmbedding(decoder_input_dim, decoder_embedding_dim)
            self.decoder_embedding_transform = torch.nn.Linear(decoder_embedding_dim, emb_dim)
        else:
            self.decoder_embedding_transform = torch.nn.Linear(decoder_input_dim, emb_dim)    

        # output layer    
        self.fc_output = nn.Linear(emb_dim, tgt_vocab_size)
        self.fc_output_reg = nn.Linear(emb_dim, tgt_reg_size)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                tgt_mask: Tensor):
        
        # encoder
        src_emb = self.encoder_positional_encoding(src) # add positional encoding to the kinematics data
        src_emb = self.encoder_input_transform(src_emb)
        if self.activation:
            src_emb = self.activation(src_emb)

        # decoder
        if self.decoder_embedding_dim > 0:
            trg = self.tgt_tok_emb(trg) # if using label encoding as well, encode the gesture inputs to the decoder
        tgt_emb = self.decoder_positional_encoding(trg) # add positional encoding to the targets (gestures)
        tgt_emb = trg
        tgt_emb = self.decoder_embedding_transform(tgt_emb) # transform tgt to model hidden dimension (d_model = emb_dim)
        if self.activation:
            tgt_emb = self.activation(tgt_emb)

        # output
        outs = self.transformer(src=src_emb, tgt=tgt_emb, src_mask=None, tgt_mask=tgt_mask)

        return self.fc_output(outs), self.fc_output_reg(outs)

    def encode(self, src: Tensor, src_mask: Tensor = None):
        x = self.encoder_input_transform(self.encoder_positional_encoding(src))
        if self.activation:
            x = self.activation(x)
        return self.transformer.encoder(x)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        if self.decoder_embedding_dim > 0:
            x = self.tgt_tok_emb(tgt)
        else:
            x = tgt
        x = self.decoder_embedding_transform(x)
        if self.activation:
            x = self.activation(x)
        return self.transformer.decoder(x, memory, tgt_mask)
    

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)