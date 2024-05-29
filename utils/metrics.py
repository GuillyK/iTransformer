import io
import itertools

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

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


def metric(pred, true, target_names):
    # mae = MAE(pred, true)
    # mse = MSE(pred, true)
    # rmse = RMSE(pred, true)
    # mape = MAPE(pred, true)
    # mspe = MSPE(pred, true)

    acc = accuracy(pred, true)
    conf_matrix = confusion_matrix_score(pred, true, target_names)
    prec = precision(pred, true)
    rec = recall(pred, true)
    F1 = f1(pred, true)

    return acc, conf_matrix, prec, rec, F1


# Metrics for time series classification
def accuracy(pred, true):
    return accuracy_score(true, pred)


def precision(pred, true):
    return precision_score(true, pred, average="micro")


def recall(pred, true):
    return recall_score(true, pred, average="micro")


def f1(pred, true):
    return f1_score(true, pred, average="micro")


def confusion_matrix_score(pred, true, class_names):
    class_preds = np.argmax(pred, axis=1)
    class_true = np.argmax(true, axis=1)
    cm = confusion_matrix(class_true, class_preds)
    figure = plot_confusion_matrix(cm, class_names)
    return figure


def plot_confusion_matrix(cm, class_names):
    figure = plt.figure(figsize=(36, 36))
    scalar_mappable = get_scalar_mappable(
        cm.flatten(), ["#de1a24", "#bf1029", "#759116", "#3f8f29", "#056517"]
    )
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=scalar_mappable.cmap,
        norm=scalar_mappable.norm,
    )
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap="Blues",
    )
    plt.title("Confusion matrix", fontsize=50)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(
        tick_marks, class_names, rotation=45, fontsize=100 / len(class_names)
    )
    plt.yticks(tick_marks, class_names, fontsize=100 / len(class_names))

    # cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    threshold = (cm.max() + cm.min()) / 2

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color=color,
            fontsize=100 / len(class_names),
        )

    plt.tight_layout()
    plt.ylabel("True label", fontsize=50)
    plt.xlabel("Predicted label", fontsize=50)

    return figure


def plot_to_image(figure):
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    plt.close(figure)
    buf.seek(0)

    digit = tf.image.decode_png(buf.getvalue(), channels=4)
    digit = tf.expand_dims(digit, 0)

    return digit
