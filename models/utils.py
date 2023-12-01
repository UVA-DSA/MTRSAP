import math
import itertools
import torch
from torch import Tensor
from torch.nn import Transformer
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



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

def plot_bars(gt, pred=None, states=None, save_path=None):

    def plot_sequence_as_horizontal_bar(sequence, title, ax):
        # if not sequence:
        #     print(f"Error: Empty sequence for {title}!")
        #     return

        # Initialize variables
        unique_elements = [sequence[0]]
        span_lengths = [1]

        # Calculate the span lengths of each element in the sequence
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                span_lengths[-1] += 1
            else:
                unique_elements.append(sequence[i])
                span_lengths.append(1)

        # Create the horizontal bar plot
        current_position = 0
        colors = "#9e0142 #d53e4f #f46d43 #fdae61 #fee08b #e6f598 #abdda4 #66c2a5 #3288bd #5e4fa2".split()
        for i in range(len(unique_elements)):
            element = unique_elements[i]
            span_length = span_lengths[i]
            ax.barh(0, span_length, left=current_position, height=1, color=colors[element])
            current_position += span_length

        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_xlabel("Sequence")
        ax.set_ylabel(title)
        ax.yaxis.label.set(rotation='horizontal', ha='right')

    def plot_difference_bar(true_sequence, pred_sequence, ax):
        # if not true_sequence or not pred_sequence:
        #     print("Error: Empty sequences!")
        #     return

        # Create a horizontal bar plot to indicate differences between sequences
        current_position = 0
        for true_elem, pred_elem in zip(true_sequence, pred_sequence):
            color = 'red' if true_elem != pred_elem else 'white'
            ax.barh(0, 1, left=current_position, height=1, color=color)
            current_position += 1

        ax.set_yticks([])
        ax.set_xticks([])
        # ax.set_title("Difference")
    
    # Replace these with your actual sequences
    true_sequence = gt
    pred_sequence = pred

    nrows = 1
    if pred is not None:
        nrows += 2 # plot the prediciton and difference bars
    if states is not None:
        nrows += 5 # plot the state changes
    fig, axes = plt.subplots(nrows=nrows, sharex=True, ncols=1, figsize=(8, 1))

    plot_sequence_as_horizontal_bar(true_sequence, "Ground Truth", axes[0])
    if pred is not None:
        plot_sequence_as_horizontal_bar(pred_sequence, "Predictions", axes[1])
        plot_difference_bar(true_sequence, pred_sequence, axes[2])
    if states is not None:
        if pred is not None:
            plot_state_changes(states, axes[3:])
        else:
            plot_state_changes(states, axes[1:])

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    # plt.show()

def plot_confusion_matrix(conf_matrix, labels):
    row_normalized_conf_matrix = conf_matrix / conf_matrix.sum(axis=1, keepdims=True)
    row_normalized_conf_matrix = np.round(row_normalized_conf_matrix, 2)

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2)
    sns.heatmap(row_normalized_conf_matrix, annot=True, cmap='Blues', xticklabels=labels, yticklabels=labels, annot_kws={'size': 12, 'ha': 'center'})
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Row-Normalized Confusion Matrix (with 2 decimal places)')
    plt.show()

def get_classification_report(pred, gt, target_names):

    # get the classification report
    labels=np.arange(0, len(target_names) ,1)
    report = classification_report(gt, pred, target_names=target_names, labels=labels, output_dict=True)
    return pd.DataFrame(report).transpose()

def merge_gesture_sequence(seq):
    import itertools
    merged_seq = list()
    for g, _ in itertools.groupby(seq): merged_seq.append(g)
    return merged_seq

def compute_edit_score(gt, pred):
    import editdistance
    max_len = max(len(gt), len(pred))
    return 1.0 - editdistance.eval(gt, pred)/max_len


def plot_state_changes(sequences, axs):

    num_sequences = len(sequences)
    # fig, axs = plt.subplots(num_sequences, 1, sharex=True, figsize=(8, 4 * num_sequences))

    markers = ['x', 'x', 'x', 'x', 'x']  # Using 'x' marker for all sequences
    labels = ['left_holding', 'left_contact', 'right_holding', 'right_contact', 'needle_state']
    colors = ['red', 'green', 'blue', 'purple', 'orange']  # Different marker colors for each sequence

    for idx, (sequence, color) in enumerate(zip(sequences, colors)):
        axs[idx].axhline(y=0, color='black')

        prev_value = None

        for i, value in enumerate(sequence):
            if prev_value is None or prev_value != value:
                axs[idx].plot(i, 0, marker=markers[idx], color=color)

            prev_value = value

        # axs[idx].set_title(f'Sequence {idx + 1}')
        axs[idx].set_ylabel(labels[idx])
        axs[idx].yaxis.label.set(rotation='horizontal', ha='right')
        axs[idx].set_yticks([])  # Remove y ticks

    axs[num_sequences - 1].set_xlabel('Index')
    plt.tight_layout()
    plt.show()

def get_tgt_mask(window_size, device):
    return Transformer.generate_square_subsequent_mask(window_size, device)

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