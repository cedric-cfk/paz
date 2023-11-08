import tensorflow as tf

keras = tf.keras
from keras.models import Model
from keras.layers import (Input, BatchNormalization, Flatten, Dense, LSTM, TimeDistributed)
from keras.applications.mobilenet import MobileNet
import random

def VVAD_LRS3_LSTM(weights=None, input_shape=(38, 96, 96, 3), seed=305865, tmp_weights_path="../../../../CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/vvad_lrs3-weights-57.hdf5"):
    """Binary Classification for videos with 2+1D CNNs.
    # Arguments
        weights: String, path to the weights file to load. TODO add weights implementation when weights are available
        input_shape: List of integers. Input shape to the model in following format: (frames, height, width, channels)
        e.g. (38, 96, 96, 3).

    # Reference
        - [A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://arxiv.org/abs/1711.11248v3)
        - [Video classification with a 3D convolutional neural network]
        (https://www.tensorflow.org/tutorials/video/video_classification#load_and_preprocess_video_data)


        Model params according to vvadlrs3.pretrained_models.getFaceImageModel().summary()
    """
    if len(input_shape) != 4:
        raise ValueError(
            '`input_shape` must be a tuple of 4 integers. '
            'Received: %s' % (input_shape,))

    # random.seed(seed)
    # initializer_glorot_lstm = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_dense = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_glorot_output = tf.keras.initializers.GlorotUniform(seed=random.randint(0, 1000000))
    # initializer_orthogonal = tf.keras.initializers.Orthogonal(seed=random.randint(0, 1000000))

    # input_shape = (None, 10, HEIGHT, WIDTH, 3)
    image = Input(shape=input_shape, name='image')
    x = image

    base_model = MobileNet(
        weights="/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/test.hdf5", include_top=False, input_shape=input_shape[1:])

    flatten = Flatten()(base_model.output)
    base_model = Model(base_model.input, flatten)
    x = TimeDistributed(base_model)(x)

    x = LSTM(32)(x)
    x = BatchNormalization()(x)

    # Add some more dense here
    for i in range(1):
        x = Dense(512, activation='relu')(x)

    x = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=image, outputs=x, name='Vvad_lrs3')

    if weights is not None:
        print("loading weights")
        model.load_weights(tmp_weights_path)  # TODO Add download link

    return model

# VVAD_LRS3_LSTM(weights="yes", tmp_weights_path="/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/vvad_lrs3-weights-57.hdf5").summary()
#
#
# import h5py
# import numpy as np
#
# dest = h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/test.hdf5", mode="w")
# src = h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/vvad_lrs3-weights-57.hdf5", mode="r")
#
# list1 = np.array([])
# # list2 = []
#
# # for layer_name in src['time_distributed']:
# #     list2.append(layer_name)
# #
# # list2.sort(key=lambda x: x.split()[0])
# # list2 = ['conv1', 'conv1_bn', 'conv_dw_1', 'conv_dw_1_bn', 'conv_dw_10', 'conv_dw_10_bn', 'conv_dw_11', 'conv_dw_11_bn', 'conv_dw_12', 'conv_dw_12_bn', 'conv_dw_13', 'conv_dw_13_bn', 'conv_dw_2', 'conv_dw_2_bn', 'conv_dw_3', 'conv_dw_3_bn', 'conv_dw_4', 'conv_dw_4_bn', 'conv_dw_5', 'conv_dw_5_bn', 'conv_dw_6', 'conv_dw_6_bn', 'conv_dw_7', 'conv_dw_7_bn', 'conv_dw_8', 'conv_dw_8_bn', 'conv_dw_9', 'conv_dw_9_bn', 'conv_pw_1', 'conv_pw_1_bn', 'conv_pw_10', 'conv_pw_10_bn', 'conv_pw_11', 'conv_pw_11_bn', 'conv_pw_12', 'conv_pw_12_bn', 'conv_pw_13', 'conv_pw_13_bn', 'conv_pw_2', 'conv_pw_2_bn', 'conv_pw_3', 'conv_pw_3_bn', 'conv_pw_4', 'conv_pw_4_bn', 'conv_pw_5', 'conv_pw_5_bn', 'conv_pw_6', 'conv_pw_6_bn', 'conv_pw_7', 'conv_pw_7_bn', 'conv_pw_8', 'conv_pw_8_bn', 'conv_pw_9', 'conv_pw_9_bn']
# #
# #
# # print(list2)
#
# for layer_name in src['time_distributed']:
#     print(layer_name)
#     for weight_name in src['time_distributed'][layer_name]:
#         src.copy(src['time_distributed'][layer_name][weight_name], dest, name=layer_name + "/" + layer_name + "/" + weight_name)
#     # print(src['time_distributed'].attrs["weight_names"])
#     # print(i + ": " + str(src['time_distributed'][i].attrs.keys()))
#
#     data = src['time_distributed'].attrs["weight_names"]
#     search = layer_name + "/"
#     indexes = [i for i, v in enumerate(data) if search in v]
#     data = [data[i] for i in range(len(data)) if i in indexes]
#     # print(name + ": ", data)
#     dest[layer_name].attrs.create("weight_names", data)
#     # print(dest[i].attrs.keys())
#     list1 = np.append(list1, bytes(layer_name, 'utf-8'))
#
# dest.attrs.create('layer_names', list1)
# dest.attrs.create('backend', "tensorflow")
# dest.attrs.create('keras_version', "2.14.0")
#
# read = h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/test.hdf5", mode="r")
# print(read.attrs["backend"])
# print(read.attrs["keras_version"])
# print(read.attrs["layer_names"])
# print(read.attrs.keys())
# for i in read:
#     print("weihgts "+i+": ", read[i].attrs["weight_names"])
# # print("------------------")
# for i in read:
#     for j in read[i]:
#         print(read[i])
#         print(j + str(read[i][j]))
#
# dest.close()
# src.close()
# read.close()
#
# # for i in src["time_distributed"]:
# #     print(i)
# #
# print("------------------")
# #
# print("backend: ", h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/mobileNetBaselineWeights.h5", mode="r").attrs["backend"])
# print("version: ", h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/mobileNetBaselineWeights.h5", mode="r").attrs["keras_version"])
# print("layers: ", h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/mobileNetBaselineWeights.h5", mode="r").attrs["layer_names"])
# print("weights: ", h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/mobileNetBaselineWeights.h5", mode="r")["conv1_bn"].attrs["weight_names"])
#
# print("conv1_bn: ", h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/mobileNetBaselineWeights.h5", mode="r")["conv1_bn"])
#
# read = h5py.File("/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/CLUSTER_OUTPUTS/VVAD_LRS3/2023_10_30-09_51_57/mobileNetBaselineWeights.h5", mode="r")
# for i in read:
#     for j in read[i]:
#         print(read[i])
#         print(j + str(read[i][j]))
#
# read.close()
#
# # Component not found in datei hdf5_format.py in der methode load_subset_weights_from_hdf5_group. Da versucht wird auf die einträge aus weight_names in der normalen gruppe (also nicht in den attrs) zuzugreifen
# # <HDF5 group "/conv1" (1 members)>
# # conv1<HDF5 group "/conv1/conv1" (1 members)>
# # <HDF5 group "/conv1_bn" (1 members)>
# # conv1_bn<HDF5 group "/conv1_bn/conv1_bn" (4 members)>
# # <HDF5 group "/conv_dw_1" (1 members)>