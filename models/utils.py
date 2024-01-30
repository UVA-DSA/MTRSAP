from .transtcn import *
from .compasstcn import *
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler, ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import pandas as pd
import time
import os
import editdistance
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def merge_gesture_sequence(seq):
    merged_seq = list()
    for g, _ in itertools.groupby(seq): merged_seq.append(g)
    return merged_seq

def get_labels(frame_wise_labels):
    labels = []

    tmp = [0]
    count = 0
    for key, group in itertools.groupby(frame_wise_labels):
        action_len = len(list(group))
        tmp.append(tmp[count] + action_len)
        count += 1
        labels.append(key)
    starts = tmp[:-1]
    ends = tmp[1:]

    return labels, starts, ends


def f_score(predicted, ground_truth, overlap):
    p_label, p_start, p_end = get_labels(predicted)
    y_label, y_start, y_end = get_labels(ground_truth)

    tp = 0
    fp = 0

    hits = np.zeros(len(y_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], y_end) - np.maximum(p_start[j], y_start)
        union = np.maximum(p_end[j], y_end) - np.minimum(p_start[j], y_start)
        IoU = (1.0 * intersection / union) * ([p_label[j] == y_label[x] for x in range(len(y_label))])
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= overlap and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1
    fn = len(y_label) - sum(hits)
    return float(tp), float(fp), float(fn)

def f1_at_X(gt, preds):
    metrics = dict()
    overlap = [.1, .25, .5] # F1 @ [10, 25, 50]
    tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
    for s in range(len(overlap)):
        tp1, fp1, fn1 = f_score(preds, gt, overlap[s])
        tp[s] += tp1
        fp[s] += fp1
        fn[s] += fn1
    for s in range(len(overlap)):
        precision = tp[s] / float(tp[s] + fp[s])
        recall = tp[s] / float(tp[s] + fn[s])

        f1_ = 2.0 * (precision * recall) / (precision + recall)
        f1_ = np.nan_to_num(f1_) * 100
        metrics[f'F1@{(int(overlap[s]*100))}'] = f1_
        
    return metrics

def reset_parameters(module):
    if isinstance(module, nn.Linear):
        module.reset_parameters()


def compute_edit_score(gt, pred):
    max_len = max(len(gt), len(pred))
    return 1.0 - editdistance.eval(gt, pred)/max_len


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



def calc_accuracy(pred, gt):
    
    pred = torch.cat(pred, dim=0)
    gt = torch.cat(gt, dim=0)

    correct_predictions = torch.sum(gt == pred)
    total_predictions = gt.numel()  # Total number of elements in the tensor

    accuracy = correct_predictions.item() / total_predictions


    print("Correct predictions:", correct_predictions.item())
    print("Total predictions:", total_predictions)
    print("Accuracy:", accuracy)
    
    return accuracy


def rolling_average(arr, window_size):
    """
    Calculate a rolling average for an array of numbers.

    Args:
        arr (list): The input array of numbers.
        window_size (int): The size of the rolling window.

    Returns:
        list: The rolling average as a list.
    """
    rolling_avg = []
    for i in range(len(arr) - window_size + 1):
        window = arr[i:i + window_size]
        avg = sum(window) / window_size
        rolling_avg.append(avg)
    return rolling_avg