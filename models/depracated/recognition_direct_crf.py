import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import torch.optim as optim

from .utils import *
from torch.optim.lr_scheduler import LambdaLR


    

class DirectCRFRecognitionModel(nn.Module):
    def __init__(self,
                 input_dim: int, # dimension of the input to the encoder
                 hidden_dim: int, # the hidden dimension of the 
                 num_layers: int,
                 vocab_size: int, # number of output classes
                 encoder_type: str, # encoder type, either `lstm` or `gru` for an rnn encoder or `transformer`
                 nhead: int = None, 
                 max_len: int = None, # maximum length of the encoder input
                 dim_feedforward: int = None,
                 emb_dim: int = -1, # input transformation dim
                 dropout: float = 0.0): # activation function of the input dim matching linear layers
        super(DirectCRFRecognitionModel, self).__init__()

        # parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim

        if encoder_type in ['lstm', 'gru']:
            # encoder input transformation to model dimension
            if emb_dim:
                self.encoder_input_transform = torch.nn.Linear(in_features=input_dim, out_features=emb_dim)
            else:
                self.emb_dim = input_dim
            # rnn encoder
            RNN = nn.LSTM if encoder_type == "lstm" else nn.GRU
            self.encoder = RNN(self.emb_dim, hidden_dim // 2, num_layers=num_layers,
                        bidirectional=True, batch_first=True, dropout=dropout)
            self.encoder_type = 'rnn'

        
        elif encoder_type == 'transformer':
            # positional encoding
            self.encoder_positional_encoding = PositionalEncoding(
                input_dim, dropout=dropout, max_len=max_len)
            self.encoder_input_transform = torch.nn.Linear(in_features=input_dim, out_features=hidden_dim)
            encoder_layers = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                             dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(
                encoder_layers, num_layers=num_layers
            )
            self.encoder_type = 'transformer'

        else:
            raise ValueError("The encoder_type argument must be chosen from [lstm, gru, transformer]")

        # dropout
        self.dropout = nn.Dropout(p=dropout)
        self.crf = CRF(hidden_dim, self.vocab_size)

    def __build_features(self, sentences):
        # masks = sentences.gt(0)
        # embeds = self.embedding(sentences.long())

        # seq_length = masks.sum(1)
        # sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        # embeds = embeds[perm_idx, :]

        # pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        # packed_output, _ = self.rnn(pack_sequence)
        # lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        # _, unperm_idx = perm_idx.sort()
        # lstm_out = lstm_out[unperm_idx, :]

        if self.encoder_type == 'rnn':
            if self.emb_dim:
                sentences = self.encoder_input_transform(sentences)
            sentences = self.dropout(sentences)
            encoder_out, _ = self.encoder(sentences)
        else:
            sentences = sentences.transpose(0, 1) # batch first to seq first
            # src transformation
            src_emb = self.encoder_positional_encoding(sentences) # add positional encoding to the kinematics data
            src_emb = self.encoder_input_transform(src_emb)

            # encoder
            encoder_out = self.encoder(src_emb)

            # transpose the transformer-encoder input and output bach to batch-first
            encoder_out = encoder_out.transpose(0, 1) # seq first to batch first
            sentences = sentences.transpose(0, 1) # seq first to batch first

        masks = torch.ones_like(sentences[:, :, 0])

        return encoder_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq = self.crf(features, masks)
        return scores, tag_seq
    

class Trainer:
    def __init__(self) -> None:
        pass

    def __eval_model(self, model, device, dataloader, desc):
        model.eval()
        with torch.no_grad():
            # eval
            losses, nums = zip(*[
                (model.loss(src, tgt[:, 1:].to(device)), len(src))
                for src, tgt, future_gesture, future_kinematics in tqdm(dataloader, desc=desc)])
            losses = [l.item() for l in losses]
            return np.sum(np.multiply(losses, nums)) / np.sum(nums)

    def __save_loss(self, losses, file_path):
        pd.DataFrame(data=losses, columns=["epoch", "batch", "train_loss", "val_loss"]).to_csv(file_path, index=False)

    def __save_model(self, model_path, model):
        torch.save(model.state_dict(), model_path)
        print("save model => {}".format(model_path))

    def train_and_evaluate(self, train_dataloader, valid_dataloader, epochs, model_dir, args, device):
        model_dir = model_dir
        if not os.path.exists(model_dir):
            os.mkdir(model_dir)

        input_dim = len(train_dataloader.dataset.get_feature_names())
        vocab_dim = len(train_dataloader.dataset.get_target_names())
        hidden_dim = args['hidden_dim']
        num_layers = args['num_layers']
        encoder_type = args['encoder_type']
        emb_dim = args['emb_dim']
        dropout = args['dropout']
        nhead = args.get('nhead', None)
        dim_feedforward = args.get('dim_feedforward', None)
        max_len = args.get('max_len', None)
        optimizer_type = args.get('optimizer_type', 'Adam')
        weight_decay = args.get('weight_decay', 0.0)
        lr = args['lr']
        model = DirectCRFRecognitionModel(
            input_dim,
            hidden_dim,
            num_layers,
            vocab_dim,
            encoder_type,
            nhead,
            max_len,
            dim_feedforward,
            emb_dim,
            dropout
        )
        model = model.to(device)

        # loss
        loss_path = os.path.join(model_dir, "loss.csv")
        losses    = pd.read_csv(loss_path).values.tolist() if args['recovery'] and os.path.exists(loss_path) else []

        # optimizer
        if optimizer_type == 'Adam':
            optimizer_cls = torch.optim.Adam
        elif optimizer_type == 'AdamW':
            optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.9 ** (epoch // 10), verbose=True)

        val_loss = 0
        best_val_loss = 1e4

        accuracies, edit_scores = [], []
        
        for epoch in range(epochs):
            train_losses = []
            # train
            model.train()
            
            for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(train_dataloader)):
    
                model.zero_grad()

                loss = model.loss(src, tgt[:, 1:])
                loss.backward()
                optimizer.step()
                # print("{:2d}/{} loss: {:5.2f}, val_loss: {:5.2f}".format(
                #     epoch+1, epochs, loss, val_loss))
                losses.append([epoch, bi, loss.item(), np.nan])
                train_losses.append(loss.item())
            # scheduler.step()    
            # def moving_average(x, w):
            #     return np.convolve(x, np.ones(w), 'valid') / w
            # plt.plot(moving_average(train_losses, 10))
            # plt.show()
            # evaluation
            val_loss = self.__eval_model(model, device, dataloader=valid_dataloader, desc="eval").item()
            print("Training Loss: ", np.mean(train_losses))
            print("Validation Loss: ", val_loss)
            accuracy, edit_score = self._compute_metrics(valid_dataloader, model)
            accuracies.append(accuracy)
            edit_scores.append(edit_score)
            # save losses
            losses[-1][-1] = val_loss
            self.__save_loss(losses, loss_path)

            # save model
            if args['save_best_val_model'] and (val_loss < best_val_loss):
                best_val_loss = val_loss
                model_path = os.path.join(model_dir, 'model.pth')
                self.__save_model(model_path, model)
                print("save model(epoch: {}) => {}".format(epoch, loss_path))
        print(f"\n\nFINAL VAL LOSS: {val_loss}")
        accuracy, edit_score = self._compute_metrics(valid_dataloader, model)
        print("\n\n\n")
        return max(accuracies), max(edit_scores), val_loss

    def _compute_metrics(self, valid_dataloader, model):
        pred, gt = list(), list()
        for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(valid_dataloader)):

            _, tag_seq = model.forward(src)
            tags = np.array(tag_seq).reshape(-1)
            pred.append(tags)

            gt_tags = tgt[:, 1:].reshape(-1).cpu()
            gt.append(gt_tags)

        pred, gt = np.concatenate(pred), np.concatenate(gt)
        print('pred: ', pred)
        print('gt: ', gt)
        accuracy, edit_score, report = get_classification_report(pred, gt, valid_dataloader.dataset.get_target_names())
        return accuracy, edit_score
        