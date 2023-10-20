# import os
import argparse
# import datetime
# import math
# import helper_functions
# from codecarbon import OfflineEmissionsTracker
from tensorflow.python.data import Dataset
import tensorflow as tf

# from paz.models.classification.cnn2Plus1 import CNN2Plus1D_Light, CNN2Plus1D_Filters, CNN2Plus1D_Layers

keras = tf.keras
from keras.losses import BinaryCrossentropy

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D, VVAD_LRS3_LSTM, MoViNet, CNN2Plus1D_Light, CNN2Plus1D_Layers, CNN2Plus1D_Filters


parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='.keras/paz/datasets',
                    help='Path to dataset directory')
parser.add_argument('--weight_path', type=str,
                    default=None,
                    help='Path to model weights')
parser.add_argument('-m', '--model', type=str,
                    default='VVAD_LRS3',
                    help='Model you want to train',
                    choices=['VVAD_LRS3', 'CNN2Plus1D', 'CNN2Plus1DLight', 'CNN2Plus1DLayers', 'CNN2Plus1DFilters', 'MoViNets', 'ViViT'])
parser.add_argument('-b', '--batch_size', type=int,
                    default=16,
                    help='Batch size for training and validation')
parser.add_argument('-e', '--epochs', type=int,
                    default=200,
                    help='Epochs for training')
parser.add_argument('-w', '--warmup', type=int,
                    default=5,
                    help='Warmup epochs for training')
parser.add_argument('-o', '--output_path', type=str,
                    default="./output/",
                    help='Path to directory for saving outputs.')
parser.add_argument('--testing', action='store_true', help='Use the test split instead of the validation split')
args = parser.parse_args()

generatorVal = VVAD_LRS3(path=args.data_path, split="val", testing=args.testing, val_split=0.1, test_split=0.1)

datasetVal = Dataset.from_generator(generatorVal, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

# Add length of dataset. This needs to be manually set because we use from generator.
datasetVal = datasetVal.apply(
    tf.data.experimental.assert_cardinality(len(generatorVal))
)

# TODO should I do the predictions in Batches?
datasetVal = datasetVal.padded_batch(1)  # args.batch_size

model = None
if args.weight_path is None:
    model = CNN2Plus1D(weights="yes", input_shape=(38, 96, 96, 3))
    # model = CNN2Plus1D_Light(weights="yes", input_shape=(38, 96, 96, 3))
    # model = CNN2Plus1D_Layers(weights="yes", input_shape=(38, 96, 96, 3))
    # model = VVAD_LRS3_LSTM(weights="yes", input_shape=(38, 96, 96, 3))

else:
    model = CNN2Plus1D(weights="yes", input_shape=(38, 96, 96, 3), tmp_weights_path=args.weight_path)
    # model = CNN2Plus1D_Light(weights="yes", input_shape=(38, 96, 96, 3), tmp_weights_path=args.weight_path)
    # model = CNN2Plus1D_Layers(weights="yes", input_shape=(38, 96, 96, 3))
    # model = VVAD_LRS3_LSTM(weights="yes", input_shape=(38, 96, 96, 3))

model.summary()

# TODO for memory usage:
# https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_memory_info

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph

if tf.config.list_physical_devices('GPU'):
    print(tf.config.experimental.get_memory_info("GPU:0"))
else:
    print("No GPU found")
model.predict(datasetVal)
if tf.config.list_physical_devices('GPU'):
    print(tf.config.experimental.get_memory_info("GPU:0"))


def get_flops(model):
    concrete = tf.function(lambda inputs: model(inputs))
    concrete_func = concrete.get_concrete_function(
        [tf.TensorSpec([1, *inputs.shape[1:]]) for inputs in model.inputs])
    frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(concrete_func)
    with tf.Graph().as_default() as graph:
        tf.graph_util.import_graph_def(graph_def, name='')
        run_meta = tf.compat.v1.RunMetadata()
        opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.compat.v1.profiler.profile(graph=graph, run_meta=run_meta, cmd="op", options=opts)
        return flops.total_float_ops



# model.summary()
# print("The FLOPs is:{}".format(get_flops(model)) ,flush=True )