from typing import List
import os
from functools import partial
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from timeit import default_timer as timer
from models.direct_transformer import DirectTransformerRecognitionModel
from models.utils import get_tgt_mask, ScheduledOptim
from utils import get_dataloaders
from models.utils import get_classification_report
from datagen import feature_names, class_names, all_class_names, state_variables



# Data Params -------------------------------------------------------------------------------------------------
tasks = ["Needle_Passing", "Suturing", "Knot_Tying"]
Features = feature_names + state_variables #kinematic features + state variable features

one_hot = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
observation_window = 30
prediction_window = 40
batch_size = 64
user_left_out = 2
cast = True
train_dataloader, valid_dataloader = get_dataloaders(tasks,
                                                     user_left_out,
                                                     observation_window,
                                                     prediction_window,
                                                     batch_size,
                                                     one_hot,
                                                     class_names = all_class_names,
                                                     feature_names = Features,
                                                     cast = cast)

print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)

# loader generator aragement: (src, src_image, tgt, future_gesture, future_kinematics)
print("Obs Kinematics Shape: ", train_dataloader.dataset[0][0].shape) 
print("Obs Target Shape: ", train_dataloader.dataset[0][2].shape)
print("Future Target Shape: ", train_dataloader.dataset[0][3].shape)
print("Future Kinematics Shape: ", train_dataloader.dataset[0][4].shape)
print("Train N Trials: ", train_dataloader.dataset.get_num_trials())
print("Train Max Length: ", train_dataloader.dataset.get_max_len())
print("Test N Trials: ", valid_dataloader.dataset.get_num_trials())
print("Test Max Length: ", valid_dataloader.dataset.get_max_len())
print("Features: ", train_dataloader.dataset.get_feature_names())

# Model Params and Initialization -------------------------------------------------------------------------------------------------
torch.manual_seed(0)
emb_size = 64
nhead = 4
ffn_hid_dim = 512
num_encoder_layers = 1
num_decoder_layers = 1
decoder_embedding_dim = 8
num_features = len(train_dataloader.dataset.get_feature_names())
num_output_classes = len(train_dataloader.dataset.get_target_names())
max_len = observation_window
print(num_features)


# recognition_transformer = RecognitionModel(encoder_input_dim=num_features,
#                                             decoder_input_dim=num_output_classes, 
#                                             num_encoder_layers=num_decoder_layers,
#                                             num_decoder_layers=num_decoder_layers,
#                                             emb_size=emb_size,
#                                             nhead=nhead,
#                                             tgt_vocab_size=num_output_classes,
#                                             max_len = max_len,
#                                             decoder_embedding_dim = decoder_embedding_dim,
#                                             dim_feedforward=ffn_hid_dim,
#                                             dropout=0.1,
#                                             activation=torch.nn.GELU())

recognition_transformer = DirectTransformerRecognitionModel(encoder_input_dim=num_features,
                                            num_encoder_layers=num_decoder_layers,
                                            emb_size=emb_size,
                                            nhead=nhead,
                                            tgt_vocab_size=num_output_classes,
                                            max_len = max_len,
                                            dim_feedforward=ffn_hid_dim,
                                            dropout=0.1)
recognition_transformer = recognition_transformer.to(device)
for p in recognition_transformer.parameters():
    if p.dim() > 1:
        torch.nn.init.xavier_uniform_(p)

# loss function -------------------------------------------------------------------------------------------------
loss_fn = torch.nn.CrossEntropyLoss()

# optimizer -------------------------------------------------------------------------------------------------
optimizer = torch.optim.Adam(recognition_transformer.parameters(), lr=2e-5, betas=(0.9, 0.98), eps=1e-9)
schd_optim = ScheduledOptim(optimizer, lr_mul=1, d_model=emb_size, n_warmup_steps=2000)


# Train / Validation Loops -------------------------------------------------------------------------------------------------
def train_epoch(model, optimizer, train_dataloader):
    model.train()
    losses = 0
    running_loss = 0.0

    for i, (src, src_image, tgt, future_gesture, future_kinematics) in enumerate(train_dataloader):

        # transpose inputs into the correct shape [seq_len, batch_size, features/classes]
        src = src.transpose(0, 1) # the srd tensor is of shape [batch_size, sequence_length, features_dim]; we transpose it to the proper dimension for the transformer model
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]
        
        # get the target mask
        tgt_mask = get_tgt_mask(observation_window, device)

        # model outputs
        # logits = model(src, tgt_input, tgt_mask)
        logits = model(src)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        if one_hot:
            tgt_comp = torch.argmax(tgt_out, dim=-1).reshape(-1)
        else:
            tgt_comp = tgt_out.reshape(-1)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_comp)
        loss.backward()

        optimizer.step()
        losses += loss.item()

        # printing statistics
        running_loss += loss.item()
        if i % 50 == 0:    # print every 50 mini-batches
            print(f'[{i + 1:5d}] loss: {running_loss / 50:.3f}')
            running_loss = 0.0

    return losses / len(list(train_dataloader))


def evaluate(model, valid_dataloader):
    model.eval()
    losses = 0
    running_loss = 0.0
    pred = []
    gt = []
    accuracy = 0
    n_batches = len(valid_dataloader)
    for src, src_image, tgt, future_gesture, future_kinematics in valid_dataloader:

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)
        tgt_input = tgt[:-1, :]

        tgt_mask = get_tgt_mask(observation_window, device)

        # logits = model(src, tgt_input, tgt_mask)
        logits = model(src)
        logits_reshaped = logits.reshape(-1, logits.shape[-1])

        tgt_out = tgt[1:, :]
        if one_hot:
            tgt_comp = torch.argmax(tgt_out, dim=-1).reshape(-1)
            tgt_reshaped = torch.argmax(tgt_out, dim=-1).reshape(-1)
        else:
            tgt_comp = tgt_out.reshape(-1)
            tgt_reshaped = tgt_out.reshape(-1)
        loss = loss_fn(logits_reshaped, tgt_comp)

        predicted_targets = torch.argmax(logits_reshaped, dim=-1).cpu().detach().numpy()
        # accuracy += np.mean(predicted_targets == tgt_reshaped.cpu().numpy())
        # print("predictions: ", predicted_targets)
        # print("ground truth: ", tgt_reshaped.cpu().numpy())
        pred.append(predicted_targets.reshape(-1))
        gt.append(tgt_reshaped.cpu().numpy().reshape(-1))

        # print(f"Valid: Accuracy for frame: {accuracy}")
        losses += loss.item()

    pred, gt = np.concatenate(pred), np.concatenate(gt)
    print(get_classification_report(pred, gt, train_dataloader.dataset.get_target_names()))
    print(f"Evaluation accuracy: {accuracy/n_batches}")
    return losses / len(list(valid_dataloader)), np.mean(pred == gt)

def cross_validation(model, optimizer, users: List[int], epochs: int):
    losses, accuracies = list(), list()
    for user in users:
        train_dataloader, valid_dataloader = get_dataloaders(
            tasks,
            user,
            observation_window,
            prediction_window,
            batch_size,
            one_hot,
            class_names = all_class_names,
            feature_names = Features,
            cast = cast)
        for epoch in range(1, epochs+1):
            start_time = timer()
            train_loss = train_epoch(model, optimizer, train_dataloader)
            end_time = timer()
            val_loss, val_accuracy = evaluate(model, valid_dataloader)
            losses.append(val_loss)
            accuracies.append(val_accuracy)
            print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))
    # save the model

    # return the main metric
    return np.mean(losses), np.mean(accuracies)

# mean_cv_loss, mean_cv_accuracy = cross_validation(recognition_transformer, optimizer, [2, 3, 4, 5, 6, 8, 9], 2)
# print(mean_cv_loss, mean_cv_accuracy)
# exit()


from timeit import default_timer as timer
NUM_EPOCHS = 10

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(recognition_transformer, optimizer, train_dataloader)
    end_time = timer()
    val_loss, val_accuracy = evaluate(recognition_transformer, valid_dataloader)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, val accuracy: {val_accuracy:.3f} "f"Epoch time = {(end_time - start_time):.3f}s"))


# function to generate output sequence using greedy algorithm
def greedy_decode(model, src, max_len, start_symbol, num_classes):

    model.eval()
    with torch.no_grad():
        if not one_hot:
            ys = torch.ones(1).fill_(start_symbol).to(torch.long).to(device).view(1, 1)
        else:
            ys = F.one_hot(ys.to(torch.int64), num_classes).to(torch.float32).to(device).unsqueeze(1)
        memory = model.encode(src.unsqueeze(1)).to(device)

        for i in range(max_len-1):
            tgt_mask = get_tgt_mask(ys.shape[0], device)
            out = model.decode(ys, memory, tgt_mask)
            prob = model.fc_output(out[-1, :, :])
            next_word = torch.argmax(prob.view(-1))
            # next_word = next_word.item()
            if not one_hot:
                new_y = next_word.to(torch.long).to(device).view(1, 1)
            else:
                new_y = F.one_hot(torch.ones(1).fill_(next_word).to(torch.int64), num_classes).to(torch.float32).reshape(1, 1, -1)
            ys = torch.cat([ys, new_y], dim=0)

    return ys

max_len = 30
X_trial, Y_trial = valid_dataloader.dataset.get_trial(0)
initial_symbol = Y_trial[-max_len-1]
X_trial, Y_trial = X_trial[-max_len:], Y_trial[-max_len:]
# pred = greedy_decode(recognition_transformer, X_trial, X_trial.shape[0], initial_symbol, len(valid_dataloader.dataset.get_target_names()))
pred = recognition_transformer(X_trial.unsqueeze(1))
print(pred.shape)
print(Y_trial)
print(torch.argmax(pred, dim=-1).view(-1))
# print(torch.argmax(pred2, dim=-1).view(-1))
# print(torch.argmax(Y_trial, dim=1))
# print(torch.mean((torch.argmax(pred, dim=-1).view(-1) == torch.argmax(Y_trial, dim=1)).to(torch.float32)))





def get_optimal_params():
    # build an objective based on the cross validation metric
    # search over the hyper parameters, find the optimal one and save it
    pass