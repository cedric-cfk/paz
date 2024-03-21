import os
from glob import glob
import metrics_functions
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from paz.models.classification.cnn2Plus1 import CNN2Plus1D

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    # 'text.usetex': True,  # Causes errors in latex
    'pgf.rcfonts': False,
    'figure.autolayout': True,
})


class Plotter:
    def __init__(self):
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/cpu16/only_normal"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/Diff_Hardware/gpu"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/Diff_Hardware"  # Only for the duration_across_hardware
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/cpu16/model_selection/Layers"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/reduction_models/cpu16/all"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/reduction_models/gpu/all"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/reduction_models/gpu/memory"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/reduction_models/cpu16/flops_and_memory"
        self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/reduction_models/cpu16/all+vvadlrs3"

        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/eval"
        # self.eval_path = "/paz/examples/visual_voice_activity_detection/evaluation/output"
        # self.eval_path = "/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/test_data/VVAD_diff_seeds/172272"

        self.model_name_dict = {'VVAD_LRS3': 'VVAD LRS3 LSTM', 'CNN2Plus1D': 'R(2+1)D',
                                'CNN2Plus1DFilters': 'R(2+1)D Filters', 'CNN2Plus1DLayers': 'R(2+1)D Layers',
                                'CNN2Plus1DLight': 'R(2+1)D Light', 'CNN2Plus1D18': 'R(2+1)D-18',
                                "CNN2Plus1DFilters_weights-21": "R(2+1)D Filters 21",
                                "CNN2Plus1DFilters_weights-23": "R(2+1)D Filters 23",
                                "CNN2Plus1DLayers_weights-15": "R(2+1)D Layers 15",
                                "CNN2Plus1DLayers_weights-17": "R(2+1)D Layers 17",
                                "CNN2Plus1DLight_weight-35": "R(2+1)D Light 35",
                                "CNN2Plus1DLight_weight-36": "R(2+1)D Light 36",
                                "CNN2Plus1DLight_weights-21": "R(2+1)D Light 21",
                                "CNN2Plus1DLight_weights-55": "R(2+1)D Light 55",
                                "CNN2Plus1D_Filters_13_cut": "R(2+1)D Filters 13 frames clipped",
                                "CNN2Plus1D_Filters_20_cut": "R(2+1)D Filters 20 frames clipped",
                                "CNN2Plus1D_Filters_13_reduce": "R(2+1)D Filters 13 frames reduced",
                                "CNN2Plus1D_Filters_20_reduce": "R(2+1)D Filters 20 frames reduced",
                                "CNN2Plus1D_Light_13_cut": "R(2+1)D Light 13 frames clipped",
                                "CNN2Plus1D_Light_20_cut": "R(2+1)D Light 20 frames clipped",
                                "CNN2Plus1D_Light_13_reduce": "R(2+1)D Light 13 frames reduced",
                                "CNN2Plus1D_Light_20_reduce": "R(2+1)D Light 20 frames reduced"}

        self.model_name_dict_small = {'VVAD_LRS3': 'Base', 'CNN2Plus1D': 'R(2+1)D',
                                      'CNN2Plus1DFilters': 'R(2+1)D Filters', 'CNN2Plus1DLayers': 'R(2+1)D Layers',
                                      'CNN2Plus1DLight': 'R(2+1)D Light', 'CNN2Plus1D18': 'R(2+1)D-18',
                                      "CNN2Plus1DFilters_weights-21": "R(2+1)D Filters 21",
                                      "CNN2Plus1DFilters_weights-23": "R(2+1)D Filters 23",
                                      "CNN2Plus1DLayers_weights-15": "R(2+1)D Layers 15",
                                      "CNN2Plus1DLayers_weights-17": "R(2+1)D Layers 17",
                                      "CNN2Plus1DLight_weight-35": "R(2+1)D Light 35",
                                      "CNN2Plus1DLight_weight-36": "R(2+1)D Light 36",
                                      "CNN2Plus1DLight_weights-21": "R(2+1)D Light 21",
                                      "CNN2Plus1DLight_weights-55": "R(2+1)D Light 55",
                                      "CNN2Plus1D_Filters_13_cut": "Filters 13 clipped",
                                      "CNN2Plus1D_Filters_20_cut": "Filters 20 clipped",
                                      "CNN2Plus1D_Filters_13_reduce": "Filters 13 reduced",
                                      "CNN2Plus1D_Filters_20_reduce": "Filters 20 reduced",
                                      "CNN2Plus1D_Light_13_cut": "Light 13 clipped",
                                      "CNN2Plus1D_Light_20_cut": "Light 20 clipped",
                                      "CNN2Plus1D_Light_13_reduce": "Light 13 reduced",
                                      "CNN2Plus1D_Light_20_reduce": "Light 20 reduced"}

        self.model_order = ['VVAD_LRS3', 'CNN2Plus1D', 'CNN2Plus1DFilters', 'CNN2Plus1DLayers', 'CNN2Plus1DLight',
                            'CNN2Plus1D18', "CNN2Plus1D_Filters_13_cut", "CNN2Plus1D_Filters_20_cut",
                            "CNN2Plus1D_Filters_13_reduce", "CNN2Plus1D_Filters_20_reduce",
                            "CNN2Plus1D_Light_13_cut",
                            "CNN2Plus1D_Light_20_cut", "CNN2Plus1D_Light_13_reduce", "CNN2Plus1D_Light_20_reduce",
                            "CNN2Plus1DFilters_weights-21", "CNN2Plus1DFilters_weights-23",
                            "CNN2Plus1DLayers_weights-15", "CNN2Plus1DLayers_weights-17", "CNN2Plus1DLight_weights-21",
                            "CNN2Plus1DLight_weight-35", "CNN2Plus1DLight_weight-36", "CNN2Plus1DLight_weights-55"]

        # NOTE: epoch number is not equal to the index in the csv files. epoch number is weight index + 1
        self.model_weights = {'VVAD_LRS3': 23, 'CNN2Plus1D': 21, 'CNN2Plus1DFilters': 21, 'CNN2Plus1DLayers': 17,
                              'CNN2Plus1DLight': 35, "CNN2Plus1DFilters_weights-21": 21,
                              "CNN2Plus1DFilters_weights-23": 23, "CNN2Plus1DLayers_weights-15": 15,
                              "CNN2Plus1DLayers_weights-17": 17, "CNN2Plus1DLight_weight-35": 35,
                              "CNN2Plus1DLight_weight-36": 36, "CNN2Plus1DLight_weights-21": 21,
                              "CNN2Plus1DLight_weights-55": 55, "CNN2Plus1D_Filters_13_cut": 2,
                              "CNN2Plus1D_Filters_20_cut": 2, "CNN2Plus1D_Filters_13_reduce": 6,
                              "CNN2Plus1D_Filters_20_reduce": 1, "CNN2Plus1D_Light_13_cut": 4,
                              "CNN2Plus1D_Light_20_cut": 2, "CNN2Plus1D_Light_13_reduce": 5,
                              "CNN2Plus1D_Light_20_reduce": 1}

        print("Loading eval data...")

        if os.path.exists(self.eval_path):
            self.model_paths = list(
                set(glob(self.eval_path + "/*/"))
                - set(glob(self.eval_path + "/plots/"))
                - set(glob(self.eval_path + "/test/"))
                - set(glob(self.eval_path + "/archive/"))
                - set(glob(self.eval_path + "/test_dataset/"))
                - set(glob(self.eval_path + "/test_reduce/"))
                - set(glob(self.eval_path + "/cpu8/"))
                - set(glob(self.eval_path + "/cpu16/"))
                - set(glob(self.eval_path + "/gpu/")))
            self.model_names = [os.path.basename(os.path.dirname(path)) for path in self.model_paths]

            self.output_path = os.path.join(self.eval_path, "plots")
            try:
                os.mkdir(self.output_path)
            except FileExistsError:
                pass
            unordered_model_names = self.model_names.copy()
            unordered_model_paths = self.model_paths.copy()
            self.model_names.sort(key=lambda name: self.model_order.index(name))

            self.model_paths.sort(key=lambda name: self.model_order.index(unordered_model_names[
                                                                              unordered_model_paths.index(name)]))

        else:
            print("Data path is not existing. Exiting...")
            exit()


    def performance_per_models(self, model_name, metric):
        if model_name == "all":
            print("Plotting all Metrics per model...")
            for model_name in self.model_names:
                self.performance_per_models(model_name, metric)
            return
        elif model_name not in self.model_names:
            print("Model not found. Chose one of the following: ", self.model_names, " Exiting...")
            return

        if metric == "all":
            print("Plotting all Metrics per model...")
            self.performance_per_models(model_name, "acc")
            self.performance_per_models(model_name, "loss")
            return
        elif metric == "acc":
            print("Plotting Accuracy per model...")
            label_name = "Accuracy"
            train_column_header = "binary_accuracy"
            val_column_header = "val_binary_accuracy"
            y_lim = (0.5, 1.0)
        elif metric == "loss":
            print("Plotting Loss per model...")
            label_name = "Loss"
            train_column_header = "loss"
            val_column_header = "val_loss"
            y_lim = (0.0, 0.9)
        else:
            print("Metric not implemented. Chose one of the following: \"acc\", \"loss\" Exiting...")
            return

        output_path = os.path.join(self.output_path, "performance_per_model")

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        matplotlib.rcParams.update({
            'font.size': 18,
        })

        i = self.model_names.index(model_name)

        train_y = metrics_functions.get_data_column(self.model_paths[i] + "training_train.log", train_column_header)
        train_y = [float(i) for i in train_y]
        val_y = metrics_functions.get_data_column(self.model_paths[i] + "training_train.log", val_column_header)
        val_y = np.array([float(i) for i in val_y])
        x = range(1, len(train_y) + 1)

        plt.figure()
        ax = plt.gca()
        plt.plot(x, train_y, label="Training Data")
        plt.plot(x, val_y, label="Validation Data")

        if label_name == "Accuracy":
            x_extreme = x[np.argmax(val_y)]
            y_extreme = val_y.max()
            arr_height = 0.7
        else:
            x_extreme = x[np.argmin(val_y)]
            y_extreme = val_y.min()
            arr_height = 0.6
            if model_name == "VVAD_LRS3":
                arr_height = 0.95

        text = "Best " + label_name + "\n x={:.0f}, y={:.3f}".format(x_extreme, y_extreme)
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->")
        kw = dict(xycoords='data', textcoords="axes fraction",
                  arrowprops=arrowprops, bbox=bbox_props, ha="center", va="top")
        ax.annotate(text, xy=(x_extreme, y_extreme), xytext=(x_extreme / len(x), arr_height), **kw)

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        # argmax = val_y.argmax()
        # plt.plot([x[argmax], x[argmax], 0], [0, val_y[argmax], val_y[argmax]],
        #          linestyle="--")

        plt.title(label_name + " for " + self.model_name_dict[self.model_names[i]])
        plt.xlabel("Epochs")
        plt.ylabel(label_name)
        plt.legend()

        plt.ylim(y_lim[0], y_lim[1])
        plt.xlim(0, len(train_y) + 1)

        # plt.xticks(plt.xticks())
        # yticks = np.append(ax.get_yticks(), [val_y.max()])
        # xticks = np.append(ax.get_xticks(), [val_y.argmax() + 1])
        # ax.set_yticks(yticks)
        # ax.set_xticks(xticks)

        plt.savefig(output_path + "/png/" + self.model_names[i] + "_" + metric + ".png")
        plt.savefig(output_path + "/pgf/" + self.model_names[i] + "_" + metric + ".pgf")
        # plt.show()


    def performance_across_models(self, metric):
        # option 1 or tow classifing if the training and test data should be plotted (1)
        # or the validation and test data (2)
        option = 2

        if metric == "all":
            print("Plotting all Metrics across models...")
            self.performance_across_models("acc")
            self.performance_across_models("loss")
            return
        elif metric == "acc":
            print("Plotting Accuracy across models...")
            label_name = "Accuracy"
            train_column_header = "binary_accuracy"
            val_column_header = "val_binary_accuracy"
            eval_column_header = "accuracy"
            y_lim = (0.7, 1.0)
        elif metric == "loss":
            print("Plotting Loss across models...")
            label_name = "Loss"
            train_column_header = "loss"
            val_column_header = "val_loss"
            eval_column_header = "loss"
            y_lim = (0.0, 0.3)
        else:
            print("Metric not implemented. Chose one of the following: \"acc\", \"loss\" Exiting...")
            return

        output_path = os.path.join(self.output_path, "performance_across_models")

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        list1 = []
        list2 = []
        for i in range(len(self.model_paths)):
            if option == 1:
                list1.append(float(
                    metrics_functions.get_data_column(self.model_paths[i] + "training_train.log", train_column_header)[
                        self.model_weights[self.model_names[i]] - 1]))
            elif option == 2:
                list1.append(float(
                    metrics_functions.get_data_column(self.model_paths[i] + "training_train.log", val_column_header)[
                        self.model_weights[self.model_names[i]] - 1]))
            list2.append(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", eval_column_header)[0]))

        print("list1:  ", list1)
        print("list2: ", list2)

        ind = np.arange(len(self.model_paths))
        width = 0.4

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()
        ax = plt.gca()
        if option == 1:
            plt.bar(ind-0.2, list1, width, label="Training Data")
        elif option == 2:
            plt.bar(ind - 0.2, list1, width, label="Validation Data")
        plt.bar(ind+0.2, list2, width, label="Testing Data")

        x = [self.model_name_dict_small[name] for name in self.model_names]
        plt.xticks(ind, x)

        plt.ylim(y_lim[0], y_lim[1])

        # TODO Maybe print the values on top of the bars?
        plt.title(label_name + " across models")
        plt.xlabel("Model Names")
        plt.ylabel(label_name)
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='center')
        if metric == "acc":
            plt.legend(loc='lower right')
        else:
            plt.legend(loc='upper right')

        if option == 1:
            plt.savefig(output_path + "/png/" + metric + "_train.png")
            plt.savefig(output_path + "/pgf/" + metric + "_train.pgf")
        elif option == 2:
            plt.savefig(output_path + "/png/" + metric + "_val.png")
            plt.savefig(output_path + "/pgf/" + metric + "_val.pgf")

        # plt.show()

    def performance_vs_efficiency_across_weights(self, performance, efficiency):
        if performance == "all":
            print("Plotting all Metrics across models...")
            self.performance_vs_efficiency_across_weights("acc", "flops")
            self.performance_vs_efficiency_across_weights("acc", "parameters")
            self.performance_vs_efficiency_across_weights("acc", "duration")
            self.performance_vs_efficiency_across_weights("loss", "flops")
            self.performance_vs_efficiency_across_weights("loss", "parameters")
            self.performance_vs_efficiency_across_weights("loss", "duration")
            return
        elif performance == "acc":
            performance_column_header = "accuracy"
            y_label = "Accuracy"
        elif performance == "loss":
            performance_column_header = "loss"
            y_label = "Loss"
        else:
            print("Metric not implemented. Chose one of the following: \"acc\", \"loss\" Exiting...")
            return

        # TODO peak memory (maybe select last(highes) or maybe select second which seems to alwayse have the lowest
        #  memory usage)
        if performance == "all":
            print("Plotting all Metrics across models...")
            self.performance_vs_efficiency_across_weights("acc", "flops")
            self.performance_vs_efficiency_across_weights("acc", "parameters")
            self.performance_vs_efficiency_across_weights("acc", "duration")
            self.performance_vs_efficiency_across_weights("loss", "flops")
            self.performance_vs_efficiency_across_weights("loss", "parameters")
            self.performance_vs_efficiency_across_weights("loss", "duration")
            return
        elif efficiency == "flops":
            efficiency_column_header = "flops"
            x_label = "FLOPS"
        elif efficiency == "parameters":
            efficiency_column_header = "parameters"
            x_label = "Parameters"
        elif efficiency == "duration":
            efficiency_column_header = "average_duration"
            x_label = "Duration in seconds"
        else:
            print("Metric not implemented. Chose one of the following: \"flops\", \"parameters\", \"duration\" Exiting...")
            return

        print("Plotting " + y_label + " vs " + x_label + " across models...")

        output_path = os.path.join(self.output_path, "performance_vs_efficiency_across_models")

        try:
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        x_axis = []
        y_axis = []
        for i in range(len(self.model_paths)):
            x_axis.append(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", efficiency_column_header)[0]))
            y_axis.append(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", performance_column_header)[0]))

        print("list1:  ", x_axis)
        print("list2: ", y_axis)

        markers = ['o', '.', 'x', '+', 'v', '^', '<', '>', 's', 'd', '1']

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()

        # TODO When using multiple Models group them by only using one plot per model
        for i in range(len(self.model_names)):
            plt.plot(x_axis[i], y_axis[i], markers[i], label=self.model_name_dict_small[self.model_names[i]])

        plt.title(y_label + " vs " + x_label + " across models")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(numpoints=1, prop={'size': 12})

        plt.savefig(output_path + "/png/" + performance + "_vs_" + efficiency + ".png")
        plt.savefig(output_path + "/pgf/" + performance + "_vs_" + efficiency + ".pgf")
        # plt.show()

    def all_metrics_across_models_table(self):
        output_path = os.path.join(self.output_path, "all_metrics_across_models")

        try:
            os.mkdir(output_path)
        except FileExistsError:
            pass

        data = {"acc": [], "loss": [], "duration": [], "flops": [], "parameters": [], "peak memory": [],
                "precision": [], "recall": [], "f1 score": []}

        for i in range(len(self.model_paths)):
            data["acc"].append(round(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "accuracy")[0]), 6))
            data["loss"].append(round(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "loss")[0]), 6))
            data["duration"].append(round(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "average_duration")[0]), 6))
            data["flops"].append(int(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "flops")[0]))
            data["parameters"].append(int(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "parameters")[0]))
            data["peak memory"].append(str(
                metrics_functions.get_data_column(self.model_paths[i] + "training_batch.log", "peak_memory")[1]))
            data["precision"].append(round(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "precision")[0]), 6))
            data["recall"].append(round(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "recall")[0]), 6))
            data["f1 score"].append(round(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", "f1_score")[0]), 6))

        df = pd.DataFrame(data, index=[self.model_name_dict[i] for i in self.model_names])
        # print("The dataframe is:\n", df)
        df.to_latex(output_path + '/table.tex', index=True)

    def all_metrics_across_models(self, metric):
        if metric == "all":
            print("Plotting all Metrics across models...")
            self.all_metrics_across_models("acc")
            self.all_metrics_across_models("loss")
            self.all_metrics_across_models("duration")
            self.all_metrics_across_models("flops")
            self.all_metrics_across_models("parameters")
            self.all_metrics_across_models("peak_memory")
            self.all_metrics_across_models("precision")
            self.all_metrics_across_models("recall")
            self.all_metrics_across_models("f1 score")
            return
        elif metric == "acc":
            print("Plotting Accuracy across models...")
            label_name = "Accuracy"
            eval_column_header = "accuracy"
            y_lim = (0.7, 1.0)
        elif metric == "loss":
            print("Plotting Loss across models...")
            label_name = "Loss"
            eval_column_header = "loss"
            y_lim = (0.0, 0.4)
        elif metric == "duration":
            print("Plotting Duration across models...")
            label_name = "Duration in seconds"
            eval_column_header = "average_duration"
            # y_lim = (0.0, 0.15) # For GPU
            y_lim = (0.0, 0.25) # For CPU16
        elif metric == "flops":
            print("Plotting FLOPS across models...")
            label_name = "FLOPs"
            eval_column_header = "flops"
            y_lim = (4000000000, 25000000000)
            # y_lim = (2000000000, 22500000000)
        elif metric == "parameters":
            print("Plotting Parameters across models...")
            label_name = "Parameters"
            eval_column_header = "parameters"
            y_lim = (0, 5000000)
        elif metric == "peak_memory":
            print("Plotting Peak Memory across models...")
            label_name = "Peak Memory"
            eval_column_header = "peak_memory"
            y_lim = (0, 125000000)
            # return  # TODO Break if memory is 'NA'
        elif metric == "precision":
            print("Plotting Precision across models...")
            label_name = "Precision"
            eval_column_header = "precision"
            y_lim = (0.7, 1.0)
        elif metric == "recall":
            print("Plotting Recall across models...")
            label_name = "Recall"
            eval_column_header = "recall"
            y_lim = (0.7, 1.0)
        elif metric == "f1 score":
            print("Plotting F1 Score across models...")
            label_name = "F1 Score"
            eval_column_header = "f1_score"
            y_lim = (0.7, 1.0)
        else:
            print("Metric not implemented. Chose one of the following: \"acc\", \"loss\" Exiting...")
            return

        output_path = os.path.join(self.output_path, "all_metrics_across_models")

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        list = []
        for i in range(len(self.model_paths)):
            if metric == "peak_memory":
                peak_memory = metrics_functions.get_data_column(self.model_paths[i] + "training_batch.log", eval_column_header)[1]
                if peak_memory == "NA":
                    print("Peak Memory is not available for ", self.model_name_dict[self.model_names[i]])
                    print("Exiting...")
                    return
                list.append(int(peak_memory))
            else:
                list.append(float(metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", eval_column_header)[0]))

        print(list)

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()
        ax = plt.gca()

        x = [self.model_name_dict_small[name] for name in self.model_names]
        x = ['Filters', 'Light', 'Filters 13', 'Filters 20', 'Light 13', 'Light 20']
        print(x)
        if len(x) == 4:
            plt.bar(x, list, color=['blue', 'purple', 'green', 'orange'])
        else:
            plt.bar(x, list, color=['red', 'blue', 'purple', 'green', 'orange', 'black'])
        plt.ylim(y_lim[0], y_lim[1])

        # TODO Maybe print the values on top of the bars?
        plt.title(label_name + " across models")
        plt.xlabel("Model Names")
        plt.ylabel(label_name)
        # plt.legend()
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='center')

        plt.savefig(output_path + "/png/" + metric + ".png")
        plt.savefig(output_path + "/pgf/" + metric + ".pgf")
        # plt.show()


    def static_bar_plot(self):
        output_path = os.path.join(self.output_path, "static_bar_plot")

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        metric = "storage"

        if metric == "Params":
            print("Plotting Parameters across models...")
            label_name = "Parameters"
            y_lim = (0, 20000000)
            x = [self.model_name_dict["CNN2Plus1D18"], self.model_name_dict["CNN2Plus1DFilters"],
                 self.model_name_dict["CNN2Plus1DLayers"], self.model_name_dict["CNN2Plus1D"],
                 self.model_name_dict["CNN2Plus1DLight"]]
            y = [int(CNN2Plus1D(architecture="CNN2Plus1D_18",input_shape=(38, 96, 96, 3)).count_params()),
                 int(CNN2Plus1D(architecture="CNN2Plus1D_Filters", input_shape=(38, 96, 96, 3)).count_params()),
                 int(CNN2Plus1D(architecture="CNN2Plus1D_Layers", input_shape=(38, 96, 96, 3)).count_params()),
                 int(CNN2Plus1D(architecture="CNN2Plus1D", input_shape=(38, 96, 96, 3)).count_params()),
                 int(CNN2Plus1D(architecture="CNN2Plus1D",input_shape=(38, 96, 96, 3)).count_params())]
        elif metric == "storage":
            print("Plotting Parameters across models...")
            label_name = "File Size"
            y_lim = (0, 20000000)
            x = [self.model_name_dict_small["VVAD_LRS3"], self.model_name_dict_small["CNN2Plus1D"],
                 self.model_name_dict_small["CNN2Plus1DFilters"], self.model_name_dict_small["CNN2Plus1DLayers"],
                 self.model_name_dict_small["CNN2Plus1DLight"]]
            y = [17848160, 1854688, 7120440, 4004080, 520640]
        else:
            return

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()
        ax = plt.gca()

        print("test")
        print(x)
        if metric == "storage":
            plt.bar(x, y, color=['blue', 'purple', 'green', 'orange'])
        else:
            plt.bar(x, y, color=['red', 'blue', 'purple', 'green', 'orange'])

        plt.ylim(y_lim[0], y_lim[1])

        # TODO Maybe print the values on top of the bars?
        plt.title(label_name + " across models")
        plt.xlabel("Model Names")
        plt.ylabel(label_name)
        # plt.legend()
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='center')

        plt.savefig(output_path + "/png/" + metric + ".png")
        plt.savefig(output_path + "/pgf/" + metric + ".pgf")
        # plt.show()


    def duration_across_hardware(self):
        output_path = os.path.join(self.output_path, "duration_across_hardware")

        label_name = "Duration in seconds"
        # y_lim = (0.0, 0.15) # For GPU
        y_lim = (0.0, 0.35)  # For CPU16

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass
        model_paths_dict = {"cpu8": [], "cpu16": [], "gpu": []}
        for folder in list(model_paths_dict.keys()):
            data_path = self.eval_path + "/" + folder
            print("Loading ", folder, " data...")

            if os.path.exists(data_path):
                multi_model_paths = list(
                    set(glob(data_path + "/*/")) - set(glob(data_path + "/plots/")) - set(glob(data_path + "/test/"))
                    - set(glob(data_path + "/test_dataset/")) - set(glob(data_path + "/test_reduce/")))
                model_names = [os.path.basename(os.path.dirname(path)) for path in multi_model_paths]

                unordered_model_names = model_names.copy()
                unordered_model_paths = multi_model_paths.copy()
                model_names.sort(key=lambda name: self.model_order.index(name))

                multi_model_paths.sort(key=lambda name: self.model_order.index(unordered_model_names[
                                                                                  unordered_model_paths.index(name)]))
                if len(self.model_names) <= len(model_names):
                    self.model_names = model_names

                model_paths_dict[folder] = multi_model_paths
            else:
                print("Data path is not existing. Exiting...")
                exit()


        data = {"cpu8": [], "cpu16": [], "gpu": [0.0]}

        for hardware in list(model_paths_dict.keys()):
            for i in range(len(model_paths_dict[hardware])):
                # print(metrics_functions.get_data_column(model_paths_dict[hardware][i] + "training_epoch.log",
                #                                         'average_duration'))
                data[hardware].append(float(
                    metrics_functions.get_data_column(model_paths_dict[hardware][i] + "training_epoch.log",
                                                      'average_duration')[0]))

        print("models:  ", self.model_names)
        print("cpu8:  ", data["cpu8"])
        print("cpu16: ", data["cpu16"])
        print("gpu: ", data["gpu"])

        ind = np.arange(len(self.model_names))
        width = 0.4

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()
        ax = plt.gca()
        plt.bar(ind - 0.2, data["cpu8"], width, label="CPU8")
        plt.bar(ind, data["cpu16"], width, label="CPU16")
        plt.bar(ind + 0.2, data["gpu"], width, label="GPU")

        x = [self.model_name_dict_small[name] for name in self.model_names]
        plt.xticks(ind, x)

        plt.ylim(y_lim[0], y_lim[1])

        # TODO Maybe print the values on top of the bars?
        plt.title(label_name + " with different hardware")
        plt.xlabel("Model Names")
        plt.ylabel(label_name)
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='center')
        # if metric == "acc":
        #     plt.legend(loc='lower right')
        # else:
        plt.legend(loc='upper left')

        plt.savefig(output_path + "/png/duration_across_hardware.png")
        plt.savefig(output_path + "/pgf/duration_across_hardware.pgf")

        # plt.show()


    def performance_across_weights(self, metric):
        # option 1 or tow classifing if the training and test data should be plotted (1)
        # or the validation and test data (2)
        option = 2

        if metric == "all":
            print("Plotting all Metrics across models...")
            self.performance_across_weights("acc")
            self.performance_across_weights("loss")
            return
        elif metric == "acc":
            print("Plotting Accuracy across models...")
            label_name = "Accuracy"
            train_column_header = "binary_accuracy"
            val_column_header = "val_binary_accuracy"
            eval_column_header = "accuracy"
            y_lim = (0.85, 1.0)
        elif metric == "loss":
            print("Plotting Loss across models...")
            label_name = "Loss"
            train_column_header = "loss"
            val_column_header = "val_loss"
            eval_column_header = "loss"
            y_lim = (0.0, 0.3)
        else:
            print("Metric not implemented. Chose one of the following: \"acc\", \"loss\" Exiting...")
            return

        output_path = os.path.join(self.output_path, "performance_across_models")

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        list1 = []
        list2 = []
        list3 = []
        for i in range(len(self.model_paths)):
            list1.append(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_train.log", train_column_header)[
                    self.model_weights[self.model_names[i]] - 1]))
            list2.append(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_train.log", val_column_header)[
                    self.model_weights[self.model_names[i]] - 1]))
            list3.append(float(
                metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", eval_column_header)[0]))

        print("list1:  ", list1)
        print("list2: ", list2)
        print("list3: ", list3)

        ind = np.arange(len(self.model_paths))
        width = 0.2

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()
        ax = plt.gca()

        plt.bar(ind-0.2, list1, width, label="Training Data")
        plt.bar(ind, list2, width, label="Validation Data")
        plt.bar(ind+0.2, list3, width, label="Testing Data")

        x = [self.model_name_dict_small[name] for name in self.model_names]
        plt.xticks(ind, x)

        plt.ylim(y_lim[0], y_lim[1])

        # TODO Maybe print the values on top of the bars?
        plt.title(label_name + " across models")
        plt.xlabel("Model Names")
        plt.ylabel(label_name)
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='center')
        if metric == "acc":
            plt.legend(loc='lower right')
        else:
            plt.legend(loc='upper right')

        if option == 1:
            plt.savefig(output_path + "/png/" + metric + "_train.png")
            plt.savefig(output_path + "/pgf/" + metric + "_train.pgf")
        elif option == 2:
            plt.savefig(output_path + "/png/" + metric + "_val.png")
            plt.savefig(output_path + "/pgf/" + metric + "_val.pgf")

        # plt.show()


    def reduction_performance_across_models(self, metric):
        # option 1 or tow classifing if the training and test data should be plotted (1)
        # or the validation and test data (2)
        option = 2

        if metric == "all":
            print("Plotting all Metrics across models...")
            self.reduction_performance_across_models("duration")
            self.reduction_performance_across_models("acc")
            # self.reduction_performance_across_models("flops")
            return
        elif metric == "duration":
            print("Plotting Duration across models...")
            label_name = "Duration in seconds"
            column_header = "average_duration"
            y_lim = (0.0, 0.05) # For GPU
            # y_lim = (0.0, 0.25)  # For CPU16
        elif metric == "acc":
            print("Plotting Accuracy per model...")
            label_name = "Accuracy"
            column_header = "accuracy"
            y_lim = (0.8, 1.0)
        # elif metric == "flops":
        #     print("Plotting FLOPS across models...")
        #     label_name = "FLOPs"
        #     eval_column_header = "flops"
        #     y_lim = (2000000000, 22500000000)
        else:
            print("Metric not implemented. Chose one of the following: \"acc\", \"loss\" Exiting...")
            return

        output_path = os.path.join(self.output_path, "reduced_performance_across_models")

        try:
            pass
            os.mkdir(output_path)
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/png")
        except FileExistsError:
            pass
        try:
            os.mkdir(output_path + "/pgf")
        except FileExistsError:
            pass

        list1 = []
        list2 = []
        for i in range(len(self.model_paths)):
            print(self.model_name_dict[self.model_names[i]])
            if self.model_name_dict[self.model_names[i]].startswith("R(2+1)D Filters"):
                list1.append(float(
                    metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", column_header)[
                        0]))
            elif self.model_name_dict[self.model_names[i]].startswith("R(2+1)D Light"):
                list2.append(float(
                    metrics_functions.get_data_column(self.model_paths[i] + "training_epoch.log", column_header)[
                        0]))
            else:
                print("Model not found.")

        print("list1:  ", list1)
        print("list2: ", list2)

        ind = np.arange(int(len(self.model_paths)/2))
        width = 0.4

        matplotlib.rcParams.update({
            'font.size': 16,
        })

        plt.figure()
        ax = plt.gca()
        plt.bar(ind-0.2, list1, width, label="R(2+1)D Filters")
        plt.bar(ind+0.2, list2, width, label="R(2+1)D Light")

        # x = [self.model_name_dict_small[name] for name in self.model_names]
        print(self.model_names)
        x = ["original", "13 clipped", "20 clipped", "13 reduced", "20 reduced"]
        plt.xticks(ind, x)

        plt.ylim(y_lim[0], y_lim[1])

        # TODO Maybe print the values on top of the bars?
        plt.title(label_name + " across models")
        plt.xlabel("Model Names")
        plt.ylabel(label_name)
        plt.setp(ax.get_xticklabels(), rotation=15, horizontalalignment='center')
        if metric == "acc":
            plt.legend(loc='lower right')
        else:
            plt.legend(loc='upper right')

        if option == 1:
            plt.savefig(output_path + "/png/" + metric + "_train.png")
            plt.savefig(output_path + "/pgf/" + metric + "_train.pgf")
        elif option == 2:
            plt.savefig(output_path + "/png/" + metric + "_val.png")
            plt.savefig(output_path + "/pgf/" + metric + "_val.pgf")

        # plt.show()

# Plotter().performance_per_models("all", "all")
# Plotter().performance_per_models("CNN2Plus1D", "acc")

# Plotter().performance_across_models("all")

# Plotter().performance_across_weights("all")

Plotter().performance_vs_efficiency_across_weights("all", "all")

# Plotter().all_metrics_across_models_table()

# Plotter().all_metrics_across_models("all")

# Plotter().static_bar_plot()

# Plotter().duration_across_hardware()

# Plotter().reduction_performance_across_models("all")
