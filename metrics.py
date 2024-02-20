import pandas as pd
import torch
import numpy as np
import editdistance
import itertools
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd

from data import trajectory_feature_names


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

def compute_edit_score(gt, pred):
    max_len = max(len(gt), len(pred))
    return 1.0 - editdistance.eval(gt, pred)/max_len

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

def get_classification_report(pred, gt, target_names):

    # get the classification report
    labels=np.arange(0, len(target_names) ,1)
    report = classification_report(gt, pred, target_names=target_names, labels=labels, output_dict=True)
    return pd.DataFrame(report).transpose()

def compute_metrics(preds, gt, regression_preds, regressoin_gt, is_train=False):

    metrics = dict()

    # frame wise accuracy
    metrics['accuracy'] = np.mean(np.array(preds) == np.array(gt))

    # edit score
    if not is_train:
        metrics['edit_score'] = compute_edit_score(merge_gesture_sequence(gt), merge_gesture_sequence(preds))

    rmse = {'RMSE_'+k: v for k, v in zip(trajectory_feature_names, np.sqrt(mean_squared_error(regressoin_gt, regression_preds, multioutput='raw_values')).reshape(-1).tolist())}
    mae = {'MAE_'+k: v for k, v in zip(trajectory_feature_names, mean_absolute_error(regressoin_gt, regression_preds, multioutput='raw_values').reshape(-1).tolist())}
    mape = {'MAPE_'+k: v for k, v in zip(trajectory_feature_names, mean_absolute_percentage_error(regressoin_gt, regression_preds, multioutput='raw_values').reshape(-1).tolist())}
    metrics = metrics | rmse | mape | mae

    # F1 @ X
    if not is_train:
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
            metrics[f'F1@{int(overlap[s]*10)}'] = f1_

    # confusion matrix

    # classification report
    report = get_classification_report(preds, gt, valid_dataloader.dataset.get_target_names())
    # metrics['report'] = report

