import argparse
import os.path
import json

import CSVLogger
import tensorflow as tf
from tensorflow.python.data import Dataset

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D, VVAD_LRS3_LSTM, MoViNet, CNN2Plus1D_Light, CNN2Plus1D_Layers, CNN2Plus1D_Filters

keras = tf.keras
from keras.losses import BinaryCrossentropy


parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='.keras/paz/datasets',
                    help='Path to dataset directory')
parser.add_argument('--weight_path', type=str,
                    default=None,
                    help='Path to model weights')
parser.add_argument('-m', '--model', type=str,
                    default='CNN2Plus1D',
                    help='Model you want to train',
                    choices=['VVAD_LRS3', 'CNN2Plus1D', 'CNN2Plus1DLight', 'CNN2Plus1DLayers', 'CNN2Plus1DFilters', 'MoViNets', 'ViViT'])
parser.add_argument('-b', '--batch_size', type=int,
                    default=16,
                    help='Batch size for training and validation')
parser.add_argument('-o', '--output_path', type=str,
                    default="./output/",
                    help='Path to directory for saving outputs.')
parser.add_argument('--testing', action='store_true', help='Use the test split instead of the validation split')
parser.add_argument('--use_multiprocessing', action='store_true', help='Use multiprocessing for data loading')
parser.add_argument('--workers', type=int, default=5, help='Number of workers for data loading')
parser.add_argument('--max_queue_size', type=int, default=15, help='Max queue size for data loading')
parser.add_argument('-c', '--cache', type=bool, default=False, help='Cache dataset in memory or not')
parser.add_argument('--reduced_frames', type=float, default=None,
                    help='Amount of frames in fps to reduce the dataset video length to. 25 is the max fps. '
                         + '0 means no reduction. (Only available for CNN2Plus1D models)')
parser.add_argument('--reduced_frames_type', type=str, default='cut',
                    help="Method used to reduce the dataset video length If 'cut' is selected, the video is cut to "
                         + "reduced_frames. If 'reduce' is selected, reduced_frames many single frames of the video"
                         + " are removed form the clip. (Only available for CNN2Plus1D models)",
                    choices=['reduce', 'cut'])
args = parser.parse_args()

try:
    os.mkdir(os.path.join(args.output_path, args.model))
except FileExistsError:
    pass

with open(os.path.join(args.output_path, args.model, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.testing:
    generatorVal = VVAD_LRS3(path=args.data_path, split="val", testing=args.testing, val_split=1.0, test_split=0.0,
                             evaluating=True, reduction_method=args.reduced_frames_type,
                             reduced_length=args.reduced_frames)
else:
    generatorVal = VVAD_LRS3(path=args.data_path, split="val", testing=args.testing, val_split=0.1, test_split=0.1,
                             evaluating=True, reduction_method=args.reduced_frames_type,
                             reduced_length=args.reduced_frames)

video_length = generatorVal.reduced_length

datasetVal = Dataset.from_generator(generatorVal, output_signature=(tf.TensorSpec(shape=(video_length, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

# Add length of dataset. This needs to be manually set because we use from generator.
datasetVal = datasetVal.apply(
    tf.data.experimental.assert_cardinality(len(generatorVal))
)

if args.cache:
    datasetVal = datasetVal.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

datasetVal = datasetVal.padded_batch(1)  # args.batch_size

model = None
if args.weight_path is None:
    if args.model == "VVAD_LRS3":
        model = VVAD_LRS3_LSTM(weights="yes", input_shape=(video_length, 96, 96, 3))
    elif args.model == "CNN2Plus1D":
        print("Special Loading of the model.")
        # model = tf.keras.models.load_model('/media/cedric/SpeedData/Uni_Seafile/Master_Thesis/paz/examples/visual_voice_activity_detection/training/cnn2plus1d_filters.h5')
        # model = tf.saved_model.load('../training/cnn2plus1d.keras')
        # model.summary()
        model = CNN2Plus1D(weights="yes", input_shape=(video_length, 96, 96, 3))
    elif args.model == "CNN2Plus1DLight":
        model = CNN2Plus1D_Light(weights="yes", input_shape=(video_length, 96, 96, 3))
    elif args.model == "CNN2Plus1DLayers":
        model = CNN2Plus1D_Layers(weights="yes", input_shape=(video_length, 96, 96, 3))
    elif args.model == "CNN2Plus1DFilters":
        model = CNN2Plus1D_Filters(weights="yes", input_shape=(video_length, 96, 96, 3))
    else:
        raise NotImplemented("Not implemented yet")
else:
    if args.model == "VVAD_LRS3":
        model = VVAD_LRS3_LSTM(weights="yes", input_shape=(video_length, 96, 96, 3), tmp_weights_path=args.weight_path)
    elif args.model == "CNN2Plus1D":
        model = CNN2Plus1D(weights="yes", input_shape=(video_length, 96, 96, 3), tmp_weights_path=args.weight_path)
    elif args.model == "CNN2Plus1DLight":
        model = CNN2Plus1D_Light(weights="yes", input_shape=(video_length, 96, 96, 3), tmp_weights_path=args.weight_path)
    elif args.model == "CNN2Plus1DLayers":
        model = CNN2Plus1D_Layers(weights="yes", input_shape=(video_length, 96, 96, 3), tmp_weights_path=args.weight_path)
    elif args.model == "CNN2Plus1DFilters":
        model = CNN2Plus1D_Filters(weights="yes", input_shape=(video_length, 96, 96, 3), tmp_weights_path=args.weight_path)
    else:
        raise NotImplemented("Not implemented yet")

from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
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


parameters = int(model.count_params())
flops = get_flops(model)
print("The number of parameters of the model is: {}".format(parameters), flush=True)
# model.summary()
# keras.utils.plot_model(model, expand_nested=True, dpi=60, show_shapes=True)
print("The needed FLOPs of the model is: {}".format(flops), flush=True)

if not tf.config.list_physical_devices('GPU'):
    print("No GPU was detected. No GPU memory usage will be logged.")

csv_logger = CSVLogger.CSVLoggerEval(args.output_path, args.model, data_generator=generatorVal, params=parameters,
                                     flops=flops)

if args.cache:
    model.predict(datasetVal,
                  max_queue_size=args.max_queue_size,
                  workers=args.workers,
                  use_multiprocessing=args.use_multiprocessing)

model.predict(datasetVal,
              callbacks=[csv_logger],
              max_queue_size=args.max_queue_size,
              workers=args.workers,
              use_multiprocessing=args.use_multiprocessing)
