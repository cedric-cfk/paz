"""
This callback is inspired by the CSVLogger callback from Keras.
reference:
    https://keras.io/api/callbacks/csv_logger/
"""
import csv
import numpy as np
import re
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf



def get_data_column(filepath, column_header) -> list:
    file = open(filepath, "r")

    csvreader = csv.reader(file)
    header = next(csvreader)

    if column_header in header:
        index = header.index(column_header)
    else:
        print("Could not find ", column_header, " in the csv file.")
        return []

    column = []
    for row in csvreader:
        column.append(row[index])

    return column

def get_true_and_prediction_labels(filepath) -> (list, list):
    column = get_data_column(filepath, "outputs")
    if not column:
        return ([], [])

    mid = int(len(column) / 2)
    i = 0
    y_true = []
    y_pred = []
    for row in column:
        if i < mid:
            y_true.append(0)
        else:
            y_true.append(1)

        score = re.findall("([+-]?((\d+(\.\d*)?)|(\.\d+))([eE][+-]\d+)?)", row)
        if len(score) != 1:
            print("illegal amount of scores per row in csv fund!")
            return ([], [])
        y_pred.append(float(score[0][0]))
        i += 1

    return y_true, y_pred

# TODO slightly different Loss because of rounding error in the export of the outputs
def get_bce_loss(y_true, y_pred):
    if len(y_true) != len(y_pred) or not y_true or not y_pred:
        print("Could not load the data. Loss will not be calculated")
        return "NA"

    bce = tf.keras.losses.BinaryCrossentropy(reduction="sum")
    return bce(y_true, y_pred).numpy()

def plot_confusion_matrix(y_true, y_pred, model_name, labels, output_path):
    cm = tf.math.confusion_matrix(y_true, y_pred)
    ax = sns.heatmap(cm, annot=True, fmt='g')
    sns.set(rc={'figure.figsize': (12, 12)})
    sns.set(font_scale=1.4)
    ax.set_title('Confusion matrix of action recognition for ' + model_name)
    ax.set_xlabel('Predicted Action')
    ax.set_ylabel('Actual Action')
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)

    plt.savefig(output_path)

def calculate_metrics(y_true, y_pred, model_name, output_path):
    if len(y_true) != len(y_pred) or not y_true or not y_pred:
        print("Could not load the data. Precision, Recall and Accuracy will not be calculated")
        return "NA", "NA", "NA", "NA"

    y_pred = tf.round(y_pred)

    labels = ["not-speaking", "speaking"]
    plot_confusion_matrix(y_true, y_pred, model_name, labels, output_path)
    cm = tf.math.confusion_matrix(y_true, y_pred)
    tp = np.diag(cm)  # Diagonal represents true positives
    precision = dict()
    recall = dict()
    f1_scor = dict()
    for i in range(len(labels)):
        col = cm[:, i]
        fp = np.sum(col) - tp[i]  # Sum of column minus true positive is false negative
        row = cm[i, :]
        fn = np.sum(row) - tp[i]  # Sum of row minus true positive, is false negative

        precision[labels[i]] = tp[i] / (tp[i] + fp)  # Precision
        recall[labels[i]] = tp[i] / (tp[i] + fn)  # Recall
        f1_scor[labels[i]] = 2 * ((precision[labels[i]] * recall[labels[i]]) / (
                    precision[labels[i]] + recall[labels[i]] + tf.keras.backend.epsilon()))
        # print("fp: ", fp)
        # print("fn: ", fn)
    # print("tp: ", tp)

    accuracy = np.sum(tp) / len(y_pred)

    return precision[labels[1]], recall[labels[1]], f1_scor[labels[1]], accuracy


def average_prediction_duration(filepath):
    column = get_data_column(filepath, "duration")

    if not column:
        print("Could not find durations in the csv file. Average duration will not be calculated")
        return "NA"

    column = [float(i) for i in column]

    mean = np.mean(column)
    # std = np.std(rows)
    # print("mean: ", mean)
    # print("std: ", std)

    suspicious = np.array(column)[column > (mean * 2)]
    suspicious = np.append(suspicious, np.array(column)[column < (mean / 2)])
    print("Prediction times that are suspicious and are ignored at the average duration: ", suspicious)

    for i in reversed(suspicious):
        column.remove(i)

    return np.mean(column)


def memory_usage(filepath):
    column = get_data_column(filepath, "peak_memory")

    if not column and np.any(column == "NA"):
        print("Could not find memory usage in the csv file. Average and Max peak memory usage will not be calculated")
        return "NA", "NA"

    column = [int(i) for i in column]

    mean = np.mean(column)
    max_mem = np.max(column)
    return mean, max_mem
