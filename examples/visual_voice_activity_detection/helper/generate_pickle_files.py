import argparse
import pickle
import h5py
import os
from vvadlrs3.sample import FeatureizedSample

parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='/media/cedric/SpeedData/Datasets/VVAD/vvadlrs3_faceImages_small.h5',
                    help='Absolute Path to dataset file')

args = parser.parse_args()

path = os.path.abspath(args.data_path)

data = h5py.File(path, mode='r')

data_train = data.get("x_train")
labels_train = data.get("y_train")

dir = os.path.dirname(path)

for i in range(len(data_train)):
    sample = FeatureizedSample()
    sample.data = data_train[i]
    sample.label = labels_train[i]
    sample.shape = data_train[i].shape
    sample.featureType = "faceImages"
    sample.k = len(data_train[i])

    output_path = dir
    if labels_train[i] == 0:
        output_path = os.path.join(output_path, "negativeSamples", f"{i}.pickle")
    elif labels_train[i] == 1:
        output_path = os.path.join(output_path, "positiveSamples", f"{i}.pickle")
    else:
        raise ValueError("Invalid label")

    sample.save(output_path)
    # pickle.dump(data_train[i], open(output_path, "wb"))

data.close()
