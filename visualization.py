import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def plot_loss(train_losses, valid_losses, loss_type: str, experiment_name, subject_id_to_exclude):
    plt.figure(figsize=(10, 8))
    epochs = list(range(1, len(train_losses) + 1))
    plt.plot(epochs, train_losses, label='Training Loss', marker='o', linestyle='-', color='blue')
    plt.plot(epochs, valid_losses, label='Test Loss', marker='o', linestyle='-', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(loss_type)
    plt.legend()
    plt.savefig(Path(f'./results/{experiment_name}/plots/{loss_type}_fig_{subject_id_to_exclude}.png'))

def plot_stacked_time_series(actual_series, predicted_series, series_names, save_path):
    num_series = len(actual_series)
    
    print(actual_series.shape, predicted_series.shape, series_names)
    # Create a figure and axes
    fig, axes = plt.subplots(nrows=num_series, ncols=1, figsize=(10, 6*num_series))
    
    # Ensure axes is a list for consistent indexing
    # if not isinstance(axes, list):
    #     print('helooooooooooooooooooooooo')
    #     axes = [axes]
    
    for i in range(num_series):
        ax = axes[i]
        actual_data = actual_series[i]
        predicted_data = predicted_series[i]
        name = series_names[i]

        # Plot actual data in blue
        ax.plot(np.arange(len(actual_data)), actual_data, label=f'Actual {name}', color='blue')

        # Plot predicted data in red
        # ax.plot(np.arange(len(predicted_data)), predicted_data, label=f'Predicted {name}', color='red')

        # Set labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
        ax.set_title(f'{name} Time Series')
        
        # Add legend
        ax.legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()

    # Show the plot (optional)
    plt.savefig(save_path)

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