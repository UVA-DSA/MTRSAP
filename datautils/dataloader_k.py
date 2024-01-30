import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
import os
import glob
from sklearn import preprocessing
from .datagen import *

class TimeSeriesDataset(Dataset):
    def __init__(self, x_set, y_set, seq_len):
        self.x, self.y = x_set, y_set
        self.seq_len = seq_len[0]

    def __len__(self):
        # return int(np.ceil(len(self.x) / float(self.seq_len))) -1
        return int(np.ceil(len(self.x) )// self.seq_len)


    def __getitem__(self, idx):
        

        # start_idx = idx * self.seq_len
        # end_idx = (idx + 1) * self.seq_len
        
        # t + 1 -- encoder input and gt output
        
        start_idx = (idx)*self.seq_len +1
        end_idx = (idx + 1)*self.seq_len + 1
        
        # t -- decoder input + 
        ystart_idx = idx*self.seq_len
        yend_idx = (idx + 1)*self.seq_len 

        # t = [''0'',  1,  2 , 3, 4]
        # y = [''G0'',<s> G1, G2, G3, G4]
        # yshifted = [<s>,G0, G1, G2, G3]
        batch_x = self.x[start_idx:end_idx]
        batch_y = self.y[start_idx:end_idx]  # [<s>,G1, G2, G3, G4]
        batch_yshifted = self.y[ystart_idx:yend_idx]
        

        # Convert NumPy arrays to PyTorch tensors
        batch_x = torch.from_numpy(batch_x)
        batch_y = torch.from_numpy(batch_y)
        batch_yshifted = torch.from_numpy(batch_yshifted)

        # Pad sequences to ensure they have the same length within the batch
        pad_len = self.seq_len - batch_x.shape[0]
        pad_lenyshifted = self.seq_len - batch_yshifted.shape[0]
        
        if pad_len > 0:
            pad_shape = (pad_len,) + batch_x.shape[1:]
            pad_shape_y = (pad_len,) + batch_y.shape[1:]
            pad_shape_yshifted = (pad_lenyshifted,) + batch_yshifted.shape[1:]

            batch_x = torch.cat([batch_x, torch.zeros(pad_shape)], dim=0)
            batch_y = torch.cat([batch_y, torch.zeros(pad_shape_y)], dim=0)
            batch_yshifted = torch.cat([batch_yshifted, torch.zeros(pad_shape_yshifted)], dim=0)

        return batch_x, batch_y, batch_yshifted,batch_yshifted

    def on_epoch_end(self):
        indices = np.arange(len(self.x))
        # np.random.shuffle(indices)
        self.x = self.x[indices]
        self.y = self.y[indices]





def generate_data(subject_id, task, features, batch_size, seq_len):    
    
    csv_path = './ProcessedDatasets/' + task
    csv_files = glob.glob(csv_path + "/*.csv")
    
    
    train_df_list = []
    test_df_list = []
    
    subject_id = f'S0{subject_id}'
    

    for file in csv_files:
        if(subject_id in file):
            test_df_list.append(pd.read_csv(file))
        else:
            train_df_list.append(pd.read_csv(file))
            

    print('Train Subject Trials: ',len(train_df_list))
    print('Test Subject Trials: ',len(test_df_list))
    
    # Concatenate all DataFrames
    train_df   = pd.concat(train_df_list, ignore_index=True)
    test_df   = pd.concat(test_df_list, ignore_index=True)

    train_df = train_df[train_df["label"]!="-"]
    test_df = test_df[test_df["label"]!="-"]
    
    lb = preprocessing.LabelBinarizer()

    train_labels= train_df.pop('label')
    train_features = train_df

    test_labels= test_df.pop('label')
    test_features = test_df

    lb.fit(train_labels)
    
    print('Gesture Classes:',lb.classes_)
    # print(train_labels.unique())
    # print(test_labels.unique())
    print('Number of Train Samples:',len(train_labels))
    print('Number of Test Samples:',len(test_labels))

    train_labels = lb.transform(train_labels)
    test_labels = lb.transform(test_labels)
    
    train_features = train_features[features]
    test_features = test_features[features]
    
    train_x = train_features.to_numpy()
    train_y = train_labels

    test_x = test_features.to_numpy()
    test_y = test_labels
    

    train_dataset = TimeSeriesDataset(train_x, train_y, seq_len)
    test_dataset = TimeSeriesDataset(test_x, test_y, seq_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    return train_dataloader, test_dataloader
 
    