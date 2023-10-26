"""
This callback is inspired by the CSVLogger callback from Keras.
reference:
    https://keras.io/api/callbacks/csv_logger/
"""
import collections
import csv
import os.path
import time
import numpy as np
import re
import seaborn as sns  # only needed for confusion matrix
import matplotlib.pyplot as plt

import tensorflow as tf
keras = tf.keras


class CSVLoggerEval(keras.callbacks.Callback):
    """Callback that streams epoch results to a CSV file.

        Supports all values that can be represented as a string,
        including 1D iterables such as `np.ndarray`.

        Example:

        ```python
        csv_logger = CSVLogger('training.log')
        model.fit(X_train, Y_train, callbacks=[csv_logger])
        ```

        Args:
            output_path: Path to a folder where the CSVs file will be stored.
            model_name: Name of the model. Used for the filename of the CSV file.
            separator: String used to separate elements in the CSV file.
            append: Boolean. True: append if file exists (useful for continuing
                training). False: overwrite existing file.
            data_generator: The data generator that is used for the evaluation.
                Used to get the id of the current sample.
        """
    def __init__(self, output_path, model_name, separator=",", append=False, data_generator=None):
        self.sep = separator
        self.batch_csv_filepath = tf.compat.path_to_str(os.path.join(output_path, model_name, "training_batch.log"))
        self.epoch_csv_filepath = tf.compat.path_to_str(os.path.join(output_path, model_name, "training_epoch.log"))
        self.conv_matrix_filepath = tf.compat.path_to_str(os.path.join(output_path, model_name, "confusion_matrix.png"))
        self.append = append
        self.writer = None
        self.keys = None
        self.append_header = True
        self.model_name = model_name
        self.data_generator = data_generator
        self.csv_file = None
        self.epoch_time_start = None
        super().__init__()

    def on_predict_begin(self, logs=None):
        if self.append:
            if tf.io.gfile.exists(self.batch_csv_filepath):
                with tf.io.gfile.GFile(self.batch_csv_filepath, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.batch_csv_filepath, mode)

    def on_predict_batch_begin(self, batch, logs=None):
        if tf.config.list_physical_devices('GPU'):
            tf.config.experimental.reset_memory_stats("GPU:0")
        self.epoch_time_start = time.time()

    def on_predict_batch_end(self, batch, logs=None):
        logs = logs or {}

        epoch_time_end = time.time()
        duration = epoch_time_end - self.epoch_time_start

        logs["duration"] = duration

        logs["sample_id"] = self.data_generator.get_index()

        if tf.config.list_physical_devices('GPU'):
            logs["peak_memory"] = tf.config.experimental.get_memory_info("GPU:0")["peak"]
        else:
            logs["peak_memory"] = "NA"

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                    isinstance(k, collections.abc.Iterable)
                    and not is_zero_dim_ndarray
            ):
                return f"\"[{', '.join(map(str, k))}]\""
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["batch_number"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"batch_number": batch})
        row_dict.update(
            (key, handle_value(logs.get(key, "NA"))) for key in self.keys
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

    def on_predict_end(self, logs=None):
        self.csv_file.close()
        self.writer = None
        self.keys = None

        if self.append:
            if tf.io.gfile.exists(self.epoch_csv_filepath):
                with tf.io.gfile.GFile(self.epoch_csv_filepath, "r") as f:
                    self.append_header = not bool(len(f.readline()))
            mode = "a"
        else:
            mode = "w"
        self.csv_file = tf.io.gfile.GFile(self.epoch_csv_filepath, mode)

        logs = logs or {}

        y_true, y_pred = self.get_true_and_prediction_labels()

        logs["loss"] = self.get_bce_loss(y_true, y_pred)

        precision, recall, f1_score, accuracy = self.calculate_metrics(y_true, y_pred)

        logs["precision"] = precision
        logs["recall"] = recall
        logs["f1_score"] = f1_score
        logs["accuracy"] = accuracy

        logs["average_duration"] = self.average_prediction_duration()

        def handle_value(k):
            is_zero_dim_ndarray = isinstance(k, np.ndarray) and k.ndim == 0
            if isinstance(k, str):
                return k
            elif (
                    isinstance(k, collections.abc.Iterable)
                    and not is_zero_dim_ndarray
            ):
                return f"\"[{', '.join(map(str, k))}]\""
            else:
                return k

        if self.keys is None:
            self.keys = sorted(logs.keys())

        if not self.writer:

            class CustomDialect(csv.excel):
                delimiter = self.sep

            fieldnames = ["epoch"] + self.keys

            self.writer = csv.DictWriter(
                self.csv_file, fieldnames=fieldnames, dialect=CustomDialect
            )
            if self.append_header:
                self.writer.writeheader()

        row_dict = collections.OrderedDict({"epoch": "0"})
        row_dict.update(
            (key, handle_value(logs.get(key, "NA"))) for key in self.keys
        )
        self.writer.writerow(row_dict)
        self.csv_file.flush()

        self.csv_file.close()
        self.writer = None

    def get_data_column(self, column_header) -> list:
        file = open(self.batch_csv_filepath, "r")

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

    def get_true_and_prediction_labels(self) -> (list, list):
        column = self.get_data_column("outputs")
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
    def get_bce_loss(self, y_true, y_pred):
        if len(y_true) != len(y_pred) or not y_true or not y_pred:
            print("Could not load the data. Loss will not be calculated")
            return "NA"

        bce = tf.keras.losses.BinaryCrossentropy(reduction="sum")
        return bce(y_true, y_pred).numpy()

    def plot_confusion_matrix(self, actual, predicted, labels):
        cm = tf.math.confusion_matrix(actual, predicted)
        ax = sns.heatmap(cm, annot=True, fmt='g')
        sns.set(rc={'figure.figsize': (12, 12)})
        sns.set(font_scale=1.4)
        ax.set_title('Confusion matrix of action recognition for ' + self.model_name)
        ax.set_xlabel('Predicted Action')
        ax.set_ylabel('Actual Action')
        ax.xaxis.set_ticklabels(labels)
        ax.yaxis.set_ticklabels(labels)

        plt.savefig(self.conv_matrix_filepath)

    def calculate_metrics(self, y_true, y_pred):
        if len(y_true) != len(y_pred) or not y_true or not y_pred:
            print("Could not load the data. Precision, Recall and Accuracy will not be calculated")
            return "NA", "NA", "NA", "NA"

        y_pred = tf.round(y_pred)

        labels = ["not-speaking", "speaking"]
        self.plot_confusion_matrix(y_true, y_pred, labels)
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

    def average_prediction_duration(self):
        column = self.get_data_column("duration")

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

    def memory_usage(self):
        column = self.get_data_column("peak_memory")

        if not column and np.any(column == "NA"):
            print("Could not find memory usage in the csv file. Average and Max peak memory usage will not be calculated")
            return "NA", "NA"

        column = [int(i) for i in column]

        mean = np.mean(column)
        max_mem = np.max(column)
        return mean, max_mem
