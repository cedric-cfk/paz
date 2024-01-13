import os
import argparse
import datetime
import json
import math
import helper_functions
from codecarbon import OfflineEmissionsTracker
from tensorflow.python.data import Dataset
import tensorflow as tf

from paz.models.classification.cnn2Plus1 import CNN2Plus1D_Light, CNN2Plus1D_Filters, CNN2Plus1D_Layers

keras = tf.keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import AdamW, SGD  # TODO test a new docker with tensorflow 2.14
from keras.optimizers.schedules import CosineDecay

from paz.datasets import VvadLrs3Dataset
from paz.models.classification import CNN2Plus1D, VVAD_LRS3_LSTM, MoViNet

parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='.keras/paz/datasets',
                    help='Path to dataset directory')
parser.add_argument('-m', '--model', type=str,
                    default='VVAD_LRS3',
                    help='Model you want to train',
                    choices=['VVAD_LRS3', 'CNN2Plus1D', 'CNN2Plus1DLight', 'CNN2Plus1DLayers', 'CNN2Plus1DFilters',
                             'MoViNets', 'ViViT'])
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
parser.add_argument('--use_multiprocessing', action='store_true', help='Use multiprocessing for data loading')
parser.add_argument('--workers', type=int, default=5, help='Number of workers for data loading')
parser.add_argument('--max_queue_size', type=int, default=5, help='Max queue size for data loading')
parser.add_argument('--seed', type=int, default=305865, help='Seed for random number generators')
parser.add_argument('--reduced_frames', type=float, default=0.0,
                    help='Amount of frames in fps to reduce the dataset video length to. 25 is the max fps. '
                         + '0 means no reduction. (Only available for CNN2Plus1D models)')
parser.add_argument('--reduced_frames_type', type=str, default='cut',
                    help="Method used to reduce the dataset video length If 'cut' is selected, the video is cut to "
                         + "reduced_frames. If 'reduce' is selected, reduced_frames many single frames of the video"
                         + " are removed form the clip. (Only available for CNN2Plus1D models)",
                    choices=['reduce', 'cut'])
parser.add_argument('--reduced_frames_tmp_weights_path', type=str, default=None,
                    help="Path to the tmp weights file used for the reduced_frames model. "
                         + "Only used when reduced_frames is set. (Only available for CNN2Plus1D models)")
parser.add_argument('-c', '--cache', action='store_true', help='Cache dataset in memory or not')

args = parser.parse_args()

if args.reduced_frames > 0:
    output_path = os.path.join(args.output_path, args.model + "_" + str(args.reduced_frames),
                               datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
else:
    output_path = os.path.join(args.output_path, args.model, datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
try:
    if args.reduced_frames > 0:
        os.mkdir(os.path.join(args.output_path, args.model + "_" + str(args.reduced_frames)))
    else:
        os.mkdir(os.path.join(args.output_path, args.model))
except FileExistsError:
    pass
try:
    os.mkdir(output_path)
except FileExistsError:
    pass
try:
    os.mkdir(os.path.join(output_path, "checkpoints"))
except FileExistsError:
    pass

with open(os.path.join(output_path, 'commandline_args.txt'), 'w') as f:
    json.dump(args.__dict__, f, indent=2)

if args.reduced_frames > 0.0:
    generatorTrain = VvadLrs3Dataset(path=args.data_path, split="train", testing=args.testing, val_split=0.1, test_split=0.1,
                               reduction_method=args.reduced_frames_type, reduced_length=args.reduced_frames)
    generatorVal = VvadLrs3Dataset(path=args.data_path, split="validation", testing=args.testing, val_split=0.1, test_split=0.1,
                             reduction_method=args.reduced_frames_type, reduced_length=args.reduced_frames)
else:
    generatorTrain = VvadLrs3Dataset(path=args.data_path, split="train", testing=args.testing, val_split=0.1, test_split=0.1)
    generatorVal = VvadLrs3Dataset(path=args.data_path, split="validation", testing=args.testing, val_split=0.1, test_split=0.1)

video_length = generatorVal.reduced_length

datasetTrain = Dataset.from_generator(generatorTrain, output_signature=(
    tf.TensorSpec(shape=(video_length, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))
datasetVal = Dataset.from_generator(generatorVal, output_signature=(
    tf.TensorSpec(shape=(video_length, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

# Add length of dataset. This needs to be manually set because we use from generator.
datasetTrain = datasetTrain.apply(
    tf.data.experimental.assert_cardinality(len(generatorTrain))
)
datasetVal = datasetVal.apply(
    tf.data.experimental.assert_cardinality(len(generatorVal))
)

datasetTrain = datasetTrain.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
datasetVal = datasetVal.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

if args.cache:
    datasetTrain = datasetTrain.padded_batch(args.batch_size)
    datasetVal = datasetVal.padded_batch(args.batch_size)

n_batches_per_epoch = len(datasetTrain)

model = None
callbacks_array = []

# Python 3.8 does not support switch case statements :(
if args.model == "VVAD_LRS3":
    model = VVAD_LRS3_LSTM(seed=args.seed)
    loss = BinaryCrossentropy()
    # optimizer = SGD(learning_rate=0.01, decay=0.01 / args.epochs)
    optimizer = SGD()
    callbacks_array.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1,
                                                                patience=10, min_lr=0.001, cooldown=2))

    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), 'TrueNegatives', 'TruePositives',
                           'FalseNegatives', 'FalsePositives'])
elif args.model.startswith("CNN2Plus1D"):
    if args.reduced_frames > 0.0:  # TODO ADD this tmp weights path
        if args.reduced_frames_tmp_weights_path is None:
            if args.model == "CNN2Plus1DLight":
                model = CNN2Plus1D_Light(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed)
            elif args.model == "CNN2Plus1DFilters":
                model = CNN2Plus1D_Filters(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed)
            elif args.model == "CNN2Plus1DLayers":
                model = CNN2Plus1D_Layers(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed)
            else:
                model = CNN2Plus1D(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed)
        else:
            if args.model == "CNN2Plus1DLight":
                model = CNN2Plus1D_Light(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed,
                                         tmp_weights_path=args.reduced_frames_tmp_weights_path)
            elif args.model == "CNN2Plus1DFilters":
                model = CNN2Plus1D_Filters(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed,
                                           tmp_weights_path=args.reduced_frames_tmp_weights_path)
            elif args.model == "CNN2Plus1DLayers":
                model = CNN2Plus1D_Layers(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed,
                                          tmp_weights_path=args.reduced_frames_tmp_weights_path)
            else:
                model = CNN2Plus1D(weights="yes", input_shape=(video_length, 96, 96, 3), seed=args.seed,
                                   tmp_weights_path=args.reduced_frames_tmp_weights_path)

    else:
        if args.model == "CNN2Plus1DLight":
            model = CNN2Plus1D_Light(seed=args.seed)
        elif args.model == "CNN2Plus1DFilters":
            model = CNN2Plus1D_Filters(seed=args.seed)
        elif args.model == "CNN2Plus1DLayers":
            model = CNN2Plus1D_Layers(seed=args.seed)
        else:
            model = CNN2Plus1D(seed=args.seed)

    loss = BinaryCrossentropy()

    if args.reduced_frames > 0.0:  # Only used then reduced_frames is set
        lr = CosineDecay(initial_learning_rate=0.0001, decay_steps=n_batches_per_epoch * (args.epochs - args.warmup),
                         alpha=0.0)
    else:
        lr = CosineDecay(initial_learning_rate=0.0, warmup_steps=n_batches_per_epoch * args.warmup, warmup_target=0.001,
                         decay_steps=n_batches_per_epoch * (args.epochs - args.warmup), alpha=0.0)

    optimizer = AdamW(learning_rate=lr)
    lr_metric = helper_functions.get_lr_metric(optimizer)
    callbacks_array.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_binary_accuracy', factor=0.1,
                                                                patience=10, min_lr=0.00001, cooldown=2))
    model.compile(loss=loss, optimizer=optimizer,
                  metrics=[lr_metric, tf.keras.metrics.BinaryAccuracy(threshold=0.5), 'TrueNegatives', 'TruePositives',
                           'FalseNegatives', 'FalsePositives'])
elif args.model == 'MoViNets':
    model = MoViNet()
    loss = BinaryCrossentropy()
    optimizer = AdamW(learning_rate=0.001)
    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])
elif args.model == 'ViViT':
    # model = ViViT()
    raise NotImplementedError
else:
    raise Exception("Model name not found")

# Checkpoint callback that saves the weights of the network every 20 epochs
callbacks_array.append(tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join(output_path, "checkpoints/weights-{epoch:02d}.hdf5"),
    verbose=1,
    save_weights_only=True,
    save_freq=n_batches_per_epoch
))

callbacks_array.append(tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join(output_path, 'tensorboard_logs'),
    update_freq='epoch'
))

if args.reduced_frames > 0.0:  # Only used then reduced_frames is set
    callbacks_array.append(keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=5))
else:
    callbacks_array.append(keras.callbacks.EarlyStopping(monitor='val_binary_accuracy', patience=15))

callbacks_array.append(helper_functions.CSVLogger(filename=os.path.join(output_path, 'outputs_csv.log')))

tracker = OfflineEmissionsTracker(project_name="VVAD", experiment_id=args.model, country_iso_code="DEU",
                                  output_dir=output_path, output_file="codecarbon",
                                  tracking_mode="process")  # gpu_ids=[0,1,2,3], on_csv_write="append/update"
tracker.start()

model.fit(x=datasetTrain,
          epochs=args.epochs,
          callbacks=callbacks_array,
          validation_data=datasetVal,
          use_multiprocessing=args.use_multiprocessing,
          workers=args.workers,
          max_queue_size=args.max_queue_size)

tracker.stop()
