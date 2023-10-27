import h5py
import os
import metrics_functions
import tensorflow as tf
from matplotlib import animation, rc
import matplotlib.pyplot as plt
import cv2
import time

# TODO besseren namen für metrics_functions finden

def print_samples(data, shape, label, output_path):
    fps = 25
    rc('animation', html='html5')
    fig = plt.figure()
    borderSize = int(shape[1] / 8)
    value = [0, 255, 0] if label else [255, 0, 0]
    images = [
        [plt.imshow(cv2.copyMakeBorder(cv2.cvtColor(features, cv2.COLOR_BGR2RGB), top=borderSize, bottom=borderSize,
                                       left=borderSize, right=borderSize, borderType=cv2.BORDER_CONSTANT, value=value),
                    animated=True)] for features in data]

    ani = animation.ArtistAnimation(fig, images, interval=(1 / fps) * 1000, blit=True,
                                        repeat_delay=1000)
    ani.save(output_path, writer='imagemagick')
    plt.close(fig)

def save_failures(log_filepath, dataset_path):
    y_true, y_pred = metrics_functions.get_true_and_prediction_labels(log_filepath)

    y_pred = tf.round(y_pred)

    wrongs = []

    for i in range(len(y_true)):
        if y_true[i] != y_pred[i]:
            wrongs.append(i)

    sample_ids = metrics_functions.get_data_column(log_filepath, "sample_id")

    for i in wrongs:
        print("Batch: ", i)
        print("Sample ID: ", sample_ids[i])

    data = h5py.File(dataset_path, mode='r')

    # x_train = data.get("x_test")
    x_train = data.get("x_train")

    output_dir = os.path.join(os.path.dirname(log_filepath), "failures")
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    # TODO add score display in frame
    # TODO batch number in filename
    for i in wrongs:
        print_samples(x_train[i],(38,96, 96, 3), bool(y_true[i]), os.path.join(output_dir, "id" + str(i) + ".gif"))

save_failures("/home/cedric/Seafile/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/eval/CNN2Plus1D/training_batch.log",
              "/home/cedric/.keras/paz/datasets/vvadlrs3_faceImages_small.h5")
# small: /home/cedric/Seafile/Uni_Seafile/Master_Thesis/paz/examples/visual_voice_activity_detection/evaluation/training.log
# big: /home/cedric/Seafile/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/Evaluation/eval/CNN2Plus1D/training_batch.log
