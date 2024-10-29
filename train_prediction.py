from pathlib import Path
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# import warnings
# warnings.filterwarnings('ignore')

from data import get_dataloaders
from data import kinematic_feature_names, trajectory_feature_names, kinematic_feature_names_jigsaws, kinematic_feature_names_jigsaws_patient_position, class_names, all_class_names, state_variables
from metrics import compute_metrics
from visualization import plot_loss, plot_bars, plot_stacked_time_series
from models import TransformerEncoderDecoderModel



# ------------------------------------ Functions ----------------------------------
def train_model(model, optimizer, criterion, train_dataloader):

    epoch_train_losses = []
    epoch_classification_losses = []
    epoch_regression_losses = []
    running_loss = 0

    model.train()
    
    preds, gt = [], []
    traj_preds, traj_gt = [], []
            
    for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(train_dataloader)):

        # transpose inputs into the correct shape [seq_len, batch_size, features/classes]
        src = src.transpose(0, 1) # the srd tensor is of shape [batch_size, sequence_length, features_dim]; we transpose it to the proper dimension for the transformer model
        tgt = tgt[:, 1:].transpose(0, 1)
        future_gesture = future_gesture.transpose(0, 1)
        future_kinematics = future_kinematics.transpose(0, 1)
        
        # get the target mask
        # tgt_mask = get_tgt_mask(train_dataloader.dataset.prediction_window, device)
        tgt_mask = None

        # model outputs
        logits, traj = model(src, tgt, tgt_mask)

        # compute loss and step the optimizer
        optimizer.zero_grad()
        if one_hot:
            gt_output_torch = torch.argmax(future_gesture, dim=-1).reshape(-1)
        else:
            gt_output_torch = future_gesture.reshape(-1)
        loss_classification = criterion[0](logits.reshape(-1, logits.shape[-1]), gt_output_torch)
        loss_regression = args['regression_loss_multiplier']*criterion[1](traj.reshape(-1, traj.shape[-1]), future_kinematics.reshape(-1, future_kinematics.shape[-1]))
        loss = loss_classification + loss_regression
        loss.backward()
        optimizer.step()

        ## store predictions and ground truth
        # classification 
        preds_ = torch.argmax(logits.reshape(-1, logits.shape[-1]), dim=-1).reshape(-1).cpu().numpy().tolist()
        if one_hot:
            gt_ = torch.argmax(future_gesture.reshape(-1, future_gesture.shape[-1]), dim=-1).reshape(-1).cpu().numpy().tolsit()
        else:
            gt_ = future_gesture.reshape(-1).cpu().numpy().tolist()
        preds += preds_
        gt += gt_

        # prediction
        reg_preds = traj.reshape(-1, traj.shape[-1]).detach().cpu().numpy()
        reg_gt = future_kinematics.reshape(-1, future_kinematics.shape[-1]).cpu().numpy()
        traj_preds.append(reg_preds)
        traj_gt.append(reg_gt)

        # store the losses
        epoch_classification_losses.append(loss_classification.item())
        epoch_regression_losses.append(loss_regression.item())
        epoch_train_losses.append(loss.item())

        running_loss += loss.item()
    traj_preds = np.concatenate(traj_preds)
    traj_gt = np.concatenate(traj_gt)

    train_metrics = compute_metrics(preds, gt, traj_preds, traj_gt, valid_dataloader.dataset.get_target_names(), is_train=True)
    return np.mean(epoch_train_losses), np.mean(epoch_classification_losses), np.mean(epoch_regression_losses), train_metrics

def eval_model(model, criterion, valid_dataloader):

    epoch_valid_losses = []
    epoch_classification_losses = []
    epoch_regression_losses = []

    model.eval()
    running_loss = 0

    preds, probs, gt = [], [], []
    traj_preds, traj_gt = [], []

    with torch.no_grad():
            
        for bi, (src, tgt, future_gesture, future_kinematics) in enumerate(tqdm(valid_dataloader)):

            # transpose inputs into the correct shape [seq_len, batch_size, features/classes]
            src = src.transpose(0, 1) # the srd tensor is of shape [batch_size, sequence_length, features_dim]; we transpose it to the proper dimension for the transformer model
            tgt = tgt[:, 1:].transpose(0, 1)
            future_gesture = future_gesture.transpose(0, 1)
            future_kinematics = future_kinematics.transpose(0, 1)
            
            # get the target mask
            # tgt_mask = get_tgt_mask(train_dataloader.dataset.prediction_window, device)
            tgt_mask = None

            # model outputs
            logits, traj = model(src, tgt, tgt_mask)

            # compute loss
            if one_hot:
                gt_output_torch = torch.argmax(future_gesture, dim=-1).reshape(-1)
            else:
                gt_output_torch = future_gesture.reshape(-1)
            loss_classification = criterion[0](logits.reshape(-1, logits.shape[-1]), gt_output_torch)
            loss_regression = args['regression_loss_multiplier']*criterion[1](traj.reshape(-1, traj.shape[-1]), future_kinematics.reshape(-1, future_kinematics.shape[-1]))
            loss = loss_classification + loss_regression

             #store the losses
            epoch_classification_losses.append(loss_classification.item())
            epoch_regression_losses.append(loss_regression.item())
            epoch_valid_losses.append(loss.item())

            # store predictions and ground truth
            preds_ = torch.argmax(logits.reshape(-1, logits.shape[-1]), dim=-1).reshape(-1).cpu().numpy().tolist()
            if one_hot:
                gt_ = torch.argmax(future_gesture.reshape(-1, future_gesture.shape[-1]), dim=-1).reshape(-1).cpu().numpy().tolsit()
            else:
                gt_ = future_gesture.reshape(-1).cpu().numpy().tolist()
            preds += preds_
            gt += gt_
            probs.append(logits.reshape(-1, logits.shape[-1]).cpu().numpy())

            # prediction
            reg_preds = traj.reshape(-1, traj.shape[-1]).cpu().numpy()
            reg_gt = future_kinematics.reshape(-1, future_kinematics.shape[-1]).cpu().numpy()
            traj_preds.append(reg_preds)
            traj_gt.append(reg_gt)

    traj_preds = np.concatenate(traj_preds)
    traj_gt = np.concatenate(traj_gt)
    print(preds[:100])
    print(gt[:100])
    print(traj_preds[:2])
    print(traj_gt[:2])

    valid_metrics = compute_metrics(preds, gt, traj_preds, traj_gt, valid_dataloader.dataset.get_target_names(), is_train=False)
    return np.mean(epoch_valid_losses), np.mean(epoch_classification_losses), np.mean(epoch_regression_losses), valid_metrics

def save_artifacts(model, train_records, valid_records, valid_dataloader):
    Path(f'./results/{experiment_name}').mkdir(parents=True, exist_ok=True)
    Path(f'./results/{experiment_name}', 'plots').mkdir(exist_ok=True) 

    # plot and save train vs. valid losses
    train_losses, train_classification_losses, train_regression_losses, train_metrics = zip(*train_records)
    valid_losses, valid_classification_losses, valid_regression_losses, valid_metrics = zip(*valid_records)
    # only keep the metrics from the last epoch
    train_metrics = train_metrics[-1]
    valid_metrics = valid_metrics[-1] 

    # plot the losses
    plot_loss(train_losses, valid_losses, "Total Loss", experiment_name, subject_id_to_exclude)
    plot_loss(train_classification_losses, valid_classification_losses, "Gesture Prediction Loss", experiment_name, subject_id_to_exclude)
    plot_loss(train_regression_losses, valid_regression_losses, "Trajectory Prediciton Loss", experiment_name, subject_id_to_exclude)
    
    # save the accuracy for the current model and subject
    print(train_metrics)
    save_df_path_train = Path(f'./results/{experiment_name}/train_metrics.csv')
    if not save_df_path_train.exists():
        train_df = pd.DataFrame.from_dict(train_metrics, orient='index').T
    else:
        train_df = pd.read_csv(save_df_path_train, index_col=0)
    train_df.loc[subject_id_to_exclude] = train_metrics
    train_df.to_csv(save_df_path_train, header=True, index=True)

    save_df_path_valid = Path(f'./results/{experiment_name}/valid_metrics.csv')
    if not save_df_path_valid.exists():
        valid_df = pd.DataFrame.from_dict(valid_metrics, orient='index').T
    else:
        valid_df = pd.read_csv(save_df_path_valid, index_col=0)
    valid_df.loc[subject_id_to_exclude] = valid_metrics
    valid_df.to_csv(save_df_path_valid, header=True, index=True)
    
    # save the model itself
    torch.save(model.state_dict(), Path(f'./results/{experiment_name}/model_{subject_id_to_exclude}.pth'))

    # plot the predictions for a sample trial from the valid set
    X, Y, Y_future, P_future = valid_dataloader.dataset.get_trial(trial_id=0, window_size=valid_dataloader.dataset.observation_window_size)
    num_batches = X.shape[0]//batch_size
    predictions, ground_truth = [], []
    traj_predictions, traj_ground_truth = [], []
    for j in range(num_batches):
        # get the data
        x = X[j*batch_size:(j+1)*batch_size].transpose(0, 1)
        y = Y[j*batch_size:(j+1)*batch_size].transpose(0, 1)
        yf = Y_future[j*batch_size:(j+1)*batch_size]
        p = P_future[j*batch_size:(j+1)*batch_size]

        # compute model outputs
        gesture_outs, trajectory_outs = model(x, y, None)

        preds = torch.argmax(gesture_outs, dim=-1).transpose(0, 1).reshape(-1).cpu().numpy().tolist()
        predictions += preds
        ground_truth += yf.reshape(-1).cpu().numpy().tolist()
        traj_predictions.append(trajectory_outs.detach().transpose(0, 1).reshape(-1, trajectory_outs.shape[-1]).cpu().numpy())
        traj_ground_truth.append(p.reshape(-1, p.shape[-1]).cpu().numpy())

    # save trial regression results
    save_path = Path(f'./results/{experiment_name}/plots/trial_traj_{subject_id_to_exclude}.png')
    trial_traj_predictions = np.concatenate(traj_predictions, axis=0)
    trial_traj_gt = np.concatenate(traj_ground_truth, axis=0)
    plot_stacked_time_series(trial_traj_predictions.T, trial_traj_gt.T, trajectory_feature_names, str(save_path))
    pd.DataFrame(data={'subject': [7]*len(predictions), 'prediction': valid_dataloader.dataset.le.inverse_transform(predictions),
                        'ground_truth': valid_dataloader.dataset.le.inverse_transform(ground_truth)}).reset_index().rename(columns={'index': 'frame'}).to_csv('outputs7.csv', index=False)

    # save trial classification results
    save_path = Path(f'./results/{experiment_name}/plots//trail_barplot_{subject_id_to_exclude}.png')
    plot_bars(ground_truth, predictions, save_path=save_path)

### -------------------------- DATA -----------------------------------------------------
tasks = ["Suturing"]
Features = kinematic_feature_names + state_variables  #kinematic features + state variable features
# Features = kinematic_feature_names_jigsaws_no_rot_ps + state_variables

one_hot = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
observation_window = 10
prediction_window = 10
batch_size = 64
cast = True
include_resnet_features = False
include_colin_features = False
include_segmentation_features = False
normalizer = '' # ('standardization', 'min-max', 'power', '')
step = 1 # 10 Hz

for subject_id_to_exclude in [2,3,4,5,6,7,8,9]:
    train_dataloader, valid_dataloader = get_dataloaders(tasks=tasks,
                                                        subject_id_to_exclude=subject_id_to_exclude,
                                                        observation_window=observation_window,
                                                        prediction_window=prediction_window,
                                                        batch_size=batch_size,
                                                        one_hot=one_hot,
                                                        class_names=class_names['Suturing'],
                                                        feature_names=Features,
                                                        trajectory_feature_names=trajectory_feature_names,
                                                        include_resnet_features=include_resnet_features,
                                                        include_segmentation_features=include_segmentation_features,
                                                        include_colin_features=include_colin_features,
                                                        cast=cast,
                                                        normalizer=normalizer,
                                                        step=step)
    # print("datasets lengths: ", len(train_dataloader.dataset), len(valid_dataloader.dataset))
    # print("X shape: ", train_dataloader.dataset.X.shape, valid_dataloader.dataset.X.shape)
    # print("Y shape: ", train_dataloader.dataset.Y.shape, valid_dataloader.dataset.Y.shape)

    # # loader generator aragement: (src, tgt, future_gesture, future_kinematics)
    # print("Obs Kinematics Shape: ", train_dataloader.dataset[0][0].shape) 
    # print("Obs Target Shape: ", train_dataloader.dataset[0][1].shape)
    # print("Future Target Shape: ", train_dataloader.dataset[0][2].shape)
    # print("Future Kinematics Shape: ", train_dataloader.dataset[0][3].shape)
    # print("Train N Trials: ", train_dataloader.dataset.get_num_trials())
    # print("Train Max Length: ", train_dataloader.dataset.get_max_len())
    # print("Test N Trials: ", valid_dataloader.dataset.get_num_trials())
    # print("Test Max Length: ", valid_dataloader.dataset.get_max_len())
    # print("Features: ", train_dataloader.dataset.get_feature_names())
    # print(train_dataloader.dataset.samples_per_trial)
    # exit()

    #------------------------------------------Build the model and the optimizer---------------------------

    # Build the Model
    model_name = 'transformer'
    args = dict(
        num_encoder_layers = 1,
        num_decoder_layers = 1,
        emb_dim = 32,
        dropout = 0.5,
        optimizer_type = 'Adam',
        weight_decay = 0.001,
        lr = 1e-4,
        nhead = 4,
        dim_feedforward = 1024,
        decoder_embedding_dim = 8,
        regression_loss_multiplier = 1500
    )

    model = TransformerEncoderDecoderModel(encoder_input_dim=len(train_dataloader.dataset.get_feature_names()),
                 decoder_input_dim=len(train_dataloader.dataset.get_target_names()),
                 num_encoder_layers=args['num_encoder_layers'],
                 num_decoder_layers=args['num_decoder_layers'],
                 emb_dim=args['emb_dim'],
                 nhead=args['nhead'],
                 tgt_vocab_size=len(train_dataloader.dataset.get_target_names()),
                 tgt_reg_size=len(trajectory_feature_names),
                 max_encoder_len=observation_window,
                 max_decoder_len=prediction_window,
                 decoder_embedding_dim=args['decoder_embedding_dim'],
                 dim_feedforward=args['dim_feedforward'],
                 dropout=args['dropout'])
    model = model.to(device)
    
    # Build the optimizer
    if args['optimizer_type'] == 'Adam':
        optimizer_cls = torch.optim.Adam
    elif args['optimizer_type'] == 'AdamW':
        optimizer_cls = torch.optim.AdamW
    optimizer = optimizer_cls(model.parameters(), lr=args['lr'], weight_decay=args['weight_decay'])
    # optimizer = ScheduledOptim(base_optimizer, args['lr'], args['emb_dim'], 1000)

    # build the criterion
    criterion = (torch.nn.CrossEntropyLoss(), torch.nn.MSELoss())

    #----------------------------------------Training Loop-------------------------------------------------
    experiment_name = 'transformer_kin_context'
    results = {}
    epochs = 10
    train_records, valid_records = [], []
    for epoch in range(epochs):
        epoch_train_loss, epoch_train_classification_loss, epoch_train_regression_loss, train_metrics = train_model(model, optimizer, criterion, train_dataloader)
        epoch_valid_loss, epoch_valid_classification_loss, epoch_valid_regression_loss, valid_metrics = eval_model(model, criterion, valid_dataloader)

        train_records.append((epoch_train_loss, epoch_train_classification_loss, epoch_train_regression_loss, train_metrics))
        valid_records.append((epoch_valid_loss, epoch_valid_classification_loss, epoch_valid_regression_loss, valid_metrics))

        print(f"\n\nTrain Results Subject {subject_id_to_exclude} Epoch {epoch}:\n", train_metrics, '\n', 'Total Loss: ', epoch_train_loss, "Classification Loss: ", epoch_train_classification_loss, "Regression Loss: ", epoch_train_regression_loss)
        print(f"\n\nValid Results Subject {subject_id_to_exclude} Epoch {epoch}:\n", valid_metrics, '\n',  'Total Loss: ', epoch_valid_loss, "Classification Loss: ", epoch_valid_classification_loss, "Regression Loss: ", epoch_valid_regression_loss)
    
    # save model, accuracy, edit_score, loss-plots for the current subject
    save_artifacts(model, train_records, valid_records, valid_dataloader)