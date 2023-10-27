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
import metrics_functions

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

        y_true, y_pred = metrics_functions.get_true_and_prediction_labels(self.batch_csv_filepath)

        logs["loss"] = metrics_functions.get_bce_loss(y_true, y_pred)

        precision, recall, f1_score, accuracy = metrics_functions.calculate_metrics(y_true, y_pred, self.model_name, self.conv_matrix_filepath)

        logs["precision"] = precision
        logs["recall"] = recall
        logs["f1_score"] = f1_score
        logs["accuracy"] = accuracy

        logs["average_duration"] = metrics_functions.average_prediction_duration(self.batch_csv_filepath)

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
