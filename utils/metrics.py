import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score,
)


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


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    acc = accuracy(pred, true)
    conf_matrix = confusion_matrix_score(pred, true)
    prec = precision(pred, true)
    rec = recall(pred, true)
    F1 = f1(pred, true)

    return acc, conf_matrix, prec, rec, F1


# Metrics for time series classification
def accuracy(pred, true):
    return accuracy_score(true, pred)


def confusion_matrix_score(pred, true):
    return confusion_matrix(true, pred)


def precision(pred, true):
    return precision_score(true, pred)


def recall(pred, true):
    return recall_score(true, pred)


def f1(pred, true):
    return f1_score(true, pred)


# print(confusion_matrix(y_test, predictions))

# # plot_confusion_matrix function is used to visualize the confusion matrix
# plot_confusion_matrix(classifier, X_test, y_test)
