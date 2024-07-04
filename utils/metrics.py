import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_fscore_support,
)
import pandas as pd

from utils.scalar_mappable import get_scalar_mappable


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(
        np.sum((true - true.mean()) ** 2)
    )


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(
        ((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0)
    )
    return (u / d).mean(-1)


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / true))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / true))


def metric(pred, true, target_names, dates):
    # mae = MAE(pred, true)
    # mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)

    acc = accuracy(pred, true)
    conf_matrix, metrics_df_per_class = confusion_matrix_score(pred, true, target_names)
    prec, prec_micro = precision(pred, true)
    rec, rec_micro = recall(pred, true)
    F1, F1_micro = f1(pred, true)

    return acc, conf_matrix, prec, prec_micro, rec, rec_micro, F1, F1_micro, metrics_df_per_class


def accuracy_over_time(preds, trues, seq_end_length_list, dates):
    acc_dict = {}
    for lengths in seq_end_length_list:

        for pred, true, length in zip(preds, trues, lengths):

            if np.array_equal(pred, true):
                correct = 1
            else:
                correct = 0
            if length.item() not in acc_dict:
                acc_dict[length.item()] = []
            acc_dict[length.item()].append(correct)
    for length, correct_list in acc_dict.items():
        acc_dict[length] = sum(correct_list) / len(correct_list)
    acc_dict = {k: acc_dict[k] for k in sorted(acc_dict)}

    # 7 day interval
    # ['2023-01-17', '2023-01-24', '2023-01-31', '2023-02-07',
    #    '2023-02-14', '2023-02-21', '2023-02-28', '2023-03-07',
    #    '2023-03-14', '2023-03-21', '2023-03-28', '2023-04-04',
    #    '2023-04-11', '2023-04-18', '2023-04-25', '2023-05-02',
    #    '2023-05-09', '2023-05-16', '2023-05-23', '2023-05-30',
    #    '2023-06-06', '2023-06-13', '2023-06-20', '2023-06-27',
    #    '2023-07-04', '2023-07-11', '2023-07-18', '2023-07-25',
    #    '2023-08-01', '2023-08-08', '2023-08-15', '2023-08-22',
    #    '2023-08-29', '2023-09-05', '2023-09-12', '2023-09-19',
    #    '2023-09-26', '2023-10-03', '2023-10-10', '2023-10-17',
    #    '2023-10-24'],
    figure = plot_accuracy(acc_dict, dates)
    return acc_dict, figure


def plot_accuracy(acc_dict, dates):
    # Unpack the dictionary into two lists
    lengths = list(acc_dict.keys())
    accuracies = list(acc_dict.values())
    print(lengths, accuracies)
    # Convert lengths to dates
    dates = dates[-len(accuracies) :]

    # Create the plot
    figure = plt.figure(figsize=(10, 6))
    plt.plot(dates, accuracies, marker="o")
    plt.title("Accuracy over time")
    plt.xlabel("Date")
    plt.ylabel("Accuracy")
    plt.grid(True)
    plt.xticks(rotation=45)  # Rotate x-axis labels for better visibility
    plt.tight_layout()  # Adjust layout for better visibility
    return figure


# Metrics for time series classification
def accuracy(pred, true):
    return accuracy_score(true, pred)


def precision(pred, true):
    return precision_score(true, pred, average="macro"), precision_score(
        true, pred, average="micro"
    )


def recall(pred, true):
    return recall_score(true, pred, average="macro"), recall_score(
        true, pred, average="micro"
    )


def f1(pred, true):
    return f1_score(true, pred, average="macro"), f1_score(
        true, pred, average="micro"
    )


def confusion_matrix_score(pred, true, class_names):
    class_preds = np.argmax(pred, axis=1)
    class_true = np.argmax(true, axis=1)
    class_names = [name.replace('class_name_late_', '') for name in class_names]
    cm = confusion_matrix(class_true, class_preds)
    figure = plot_confusion_matrix(cm, class_names)
    precision, recall, f1, support = precision_recall_fscore_support(
        true, pred, average=None
    )
    accuracy = cm.diagonal() / cm.sum(axis=1)
    metrics_df = pd.DataFrame(
        {
            "Class": class_names,
            "Precision": precision,
            "Recall": recall,
            "F1-score": f1,
            "Support": support,
            "Accuracy": accuracy,
        }
    )
    return figure, metrics_df


def plot_confusion_matrix(cm, class_names):
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    # Set up the figure
    if len(class_names) > 20:
        plt.figure(figsize=(32, 30))
        factor = 2.5
    else:
        plt.figure(figsize=(12, 10))
        factor = 1

    # Use a diverging color palette
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.75},
        vmin=0,
        vmax=1,
    )

    # Improve labels and title
    plt.title("Confusion Matrix", fontsize=16*factor, pad=20)
    plt.xlabel("Predicted Class", fontsize=14*factor, labelpad=10)
    plt.ylabel("True Class", fontsize=14*factor, labelpad=10)

    # Adjust tick labels
    plt.xticks(rotation=45, ha="right", fontsize=10*factor)
    plt.yticks(rotation=0, fontsize=10*factor)

    # Add class names
    tick_marks = np.arange(len(class_names)) + 0.5
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Improve color bar
    cbar = plt.gcf().axes[-1]
    cbar.set_ylabel(
        "Classification Rate", rotation=270, labelpad=20, fontsize=12*factor
    )

    plt.tight_layout()
    return plt.gcf()

    # figure = plt.figure(figsize=(36, 36))
    # # scalar_mappable = get_scalar_mappable(
    # #     cm.flatten(), ["#de1a24", "#bf1029", "#759116", "#3f8f29", "#056517"]
    # # )
    # # plt.imshow(
    # #     cm,
    # #     interpolation="nearest",
    # #     cmap=scalar_mappable.cmap,
    # #     norm=scalar_mappable.norm,
    # # )
    #     # Normalize the confusion matrix to get percentages
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # plt.imshow(cm, interpolation="nearest", cmap="Blues")
    # plt.title("Confusion matrix", fontsize=50)
    # plt.colorbar(format='%.2f')  # Display two decimal places
    # tick_marks = np.arange(len(class_names))
    # plt.xticks(
    #     tick_marks, class_names, rotation=45, fontsize=100 / len(class_names)
    # )
    # plt.yticks(tick_marks, class_names, fontsize=100 / len(class_names))

    # threshold = (cm.max() + cm.min()) / 2

    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     color = "white" if cm[i, j] > threshold else "black"
    #     plt.text(
    #         j,
    #         i,
    #         f'{cm[i, j]:.2f}',  # Display two decimal places
    #         horizontalalignment="center",
    #         color=color,
    #         fontsize=100 / len(class_names),
    #     )

    # plt.tight_layout()
    # plt.ylabel("True label", fontsize=50)
    # plt.xlabel("Predicted label", fontsize=50)

    # return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit

    return digit


def visualize_attention(attention_weights, input_tokens, writer, global_step):
    """
    Visualizes the attention weights for each head.

    Parameters:
    - attention_weights: The attention weights tensor of shape (batch_size, num_heads, seq_len, seq_len).
    - input_tokens: List of input tokens.
    """
    input_tokens = list(range(0, attention_weights[0].shape[3]))
    for head in range(attention_weights[0].shape[1]):
        attn = attention_weights[0][0, head].detach().cpu().numpy()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            attn,
            xticklabels=input_tokens,
            yticklabels=input_tokens,
            cmap="viridis",
        )
        plt.xlabel("Input Tokens")
        plt.ylabel("Output Tokens")
        plt.title(f"Attention Heatmap ( Head {head})")

        writer.add_figure(
            f"Attention_Heatmap__Head_{head}", plt.gcf(), global_step
        )
        plt.close()
