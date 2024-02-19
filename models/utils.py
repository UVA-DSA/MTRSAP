import time
import os

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pandas as pd

from torch.nn import Transformer

from .transtcn import *
from .compasstcn import *
from metrics import compute_edit_score, f1_at_X

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)
    

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)
    

class ScheduledOptim():
    ''' A simple wrapper class for learning rate scheduling
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
        sched = ScheduledOptim(optimizer, d_model=..., n_warmup_steps=...)
    '''

    def __init__(self, optimizer, lr_mul, d_model, n_warmup_steps):
        self._optimizer = optimizer
        self.lr_mul = lr_mul
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.n_steps = 0


    def step(self):
        "Step with the inner optimizer"
        self._update_learning_rate()
        self._optimizer.step()


    def zero_grad(self):
        "Zero out the gradients with the inner optimizer"
        self._optimizer.zero_grad()


    def _get_lr_scale(self):
        d_model = self.d_model
        n_steps, n_warmup_steps = self.n_steps, self.n_warmup_steps
        return (d_model ** -0.5) * min(n_steps ** (-0.5), n_steps * n_warmup_steps ** (-1.5))


    def _update_learning_rate(self):
        ''' Learning rate scheduling per step '''

        self.n_steps += 1
        lr = self.lr_mul * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)

def reset_parameters(module):
    if isinstance(module, nn.Linear):
        module.reset_parameters()

def initiate_model(input_dim, output_dim, transformer_params, learning_params, tcn_model_params, model_name):

    d_model, nhead, num_layers, hidden_dim, layer_dim, encoder_params, decoder_params = transformer_params.values()

    lr, epochs, weight_decay, patience = learning_params.values()

    if (model_name == 'transformer'):
        print("Creating Transformer")
        model = TransformerModel(input_dim=input_dim, output_dim=output_dim, d_model=d_model, nhead=nhead, num_layers=num_layers,
                                 hidden_dim=hidden_dim, layer_dim=layer_dim, encoder_params=encoder_params, decoder_params=decoder_params)

    elif (model_name == 'tcn'):
        print("Creating TCN")
        model = TCN(input_dim=input_dim, output_dim=output_dim,
                    tcn_model_params=tcn_model_params)

    model = model.cuda()

    # Define the optimizer (Adam optimizer with weight decay)
    optimizer = optim.Adam(model.parameters(), lr=lr,
                           weight_decay=weight_decay, betas=(0.9,0.98), eps=1e-9)


    # Define the learning rate scheduler (ReduceLROnPlateau scheduler)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=patience, verbose=True)

    criterion = nn.CrossEntropyLoss()

    return model, optimizer, scheduler, criterion

# given a tensor of shape [batch,seq_len,features], finds the most common class within the sequence and returns [batch,features]
def find_mostcommon(tensor, device):

    batch_y_bin = torch.mode(tensor, dim=1).values
    batch_y_bin = batch_y_bin.to(device)

    return batch_y_bin

# evaluation loop (supports both window wise and frame wise)
def eval_loop(model, test_dataloader, criterion, dataloader):
    model.eval()
    
    
    with torch.no_grad():
        # eval
        losses = []
        ypreds, gts = [], []
        accuracy = 0
        n_batches = len(test_dataloader)
        
        inference_times = []
        nypreds,ngts = [],[]
        
        for src, tgt, future_gesture, future_kinematics in test_dataloader:
            
            if(dataloader == "kw"):
                src = src.to(torch.float32)
                src = src.to(device)
                
                tgt = tgt.to(torch.float32)
                tgt = tgt.to(device)  
                
            y = find_mostcommon(tgt, device) #maxpool
            # y = tgt


            start_time = time.time_ns()
            y_pred = model(src)  # [64,10]
            end_time = time.time_ns()
            inference_time = (end_time-start_time)/1e6
            inference_time = inference_time/src.shape[0] #divide by batch size to get time for single window
            inference_times.append(inference_time)

            pred = torch.argmax(y_pred, dim=-1)
            gt = torch.argmax(y, dim=-1)  # maxpool


            pred = pred.cpu().numpy()
            gt = gt.cpu().numpy()

            # print(gt,pred)
            ypreds.append(pred)
            gts.append(gt)

            loss = criterion(y_pred, y) # maxpool

            losses.append(loss.item())
            
            accuracy += np.mean(pred == gt)


        ypreds = np.concatenate(ypreds)
        gts = np.concatenate(gts)
        
        edit_distance = compute_edit_score(gts, ypreds)
        f1_score = f1_at_X(gts,ypreds)


        accuracy = accuracy/n_batches
        inference_time = np.mean(inference_times)
        print("Accuracy:", accuracy, 'Edit Score:',edit_distance, 'F1@X:',f1_score,'Inference Time per window:',inference_time)


        return np.mean(losses), accuracy, inference_time,  ypreds, gts, edit_distance, f1_score

# train loop, calls evaluation every epoch
def traintest_loop(train_dataloader, test_dataloader, model, optimizer, scheduler, criterion, epochs, dataloader, subject, modality):


    accuracy = 0
    total_accuracy = []
    
    ypreds, gts = [],[]
    highest_acc = 0
    highestypreds, highestygts = [],[]
    
    file_path = f'./model_weights/Modality_M{modality}_S0{subject}_best_model_weights.pth'
    
    # training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()

            if(dataloader == "kw"):
                src = src.to(torch.float32)
                src = src.to(device)
                
                tgt = tgt.to(torch.float32)
                tgt = tgt.to(device)  
             

            y = find_mostcommon(tgt, device) #maxpool
            # y = tgt
   
            y_pred =  model(src)  # [64,10]
            # print('input, prediction, yseq, gt:',src.shape, y_pred.shape,  y.shape, tgt.shape)
            # input()
            
  
            loss = criterion(y_pred, y)  # for maxpool
            # loss = criterion(y_pred, tgt)
            loss.backward()

            running_loss += loss.item()
            
            optimizer.step()
            
        scheduler.step(running_loss)

        print(
            f"Training Epoch {epoch+1}, Training Loss: {running_loss / len(train_dataloader):.6f}")

        # evaluation loop
        val_loss, accuracy, inference_time, ypreds, gts, edit_distance, f1_score = eval_loop(model, test_dataloader, criterion, dataloader)
        print(f"Valdiation Epoch {epoch+1}, Validation Loss: {val_loss:.6f}")

        if(accuracy > highest_acc): # save only if the accuracy is higher than before
            highest_acc = accuracy
            highestygts = gts
            highestypreds = ypreds
            # Save the model weights to the file
            # torch.save(model.state_dict(), file_path)


        total_accuracy.append(accuracy)
    
    results = {'subject':subject, 'prediction':highestygts, 'groundtruth':highestypreds}
    df = pd.DataFrame(results)

    df_outpath = './results/model_outputs/'
        # Create the directory if it doesn't exist
    if not os.path.exists(df_outpath):
        os.makedirs(df_outpath)
        print(f"Directory '{df_outpath}' created.")
    else:
        print(f"Directory '{df_outpath}' already exists.")

    df.to_csv(f'{df_outpath}S0{subject}_output.csv')
    
    return val_loss, accuracy, total_accuracy, inference_time, edit_distance, f1_score



