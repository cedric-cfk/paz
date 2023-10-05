import os
import argparse
import datetime
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

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D, VVAD_LRS3_LSTM

parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='.keras/paz/datasets',
                    help='Path to dataset directory')
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
parser.add_argument('--use_multiprocessing', action='store_true', help='Use multiprocessing for data loading')
parser.add_argument('--workers', type=int, default=5, help='Number of workers for data loading')
parser.add_argument('--max_queue_size', type=int, default=5, help='Max queue size for data loading')
parser.add_argument('--seed', type=int, default=305865, help='Seed for random number generators')

args = parser.parse_args()

output_path = os.path.join(args.output_path, args.model, datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S"))
try:
    os.mkdir(os.path.join(args.output_path, args.model))
except FileExistsError:
    pass
try:
    os.mkdir(os.path.join(output_path))
except FileExistsError:
    pass
try:
    os.mkdir(os.path.join(output_path, "checkpoints"))
except FileExistsError:
    pass

generatorTrain = VVAD_LRS3(path=args.data_path, split="train", testing=args.testing, val_split=0.1, test_split=0.1)
generatorVal = VVAD_LRS3(path=args.data_path, split="val", testing=args.testing, val_split=0.1, test_split=0.1)

datasetTrain = Dataset.from_generator(generatorTrain, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))
datasetVal = Dataset.from_generator(generatorVal, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

# Add length of dataset. This needs to be manually set because we use from generator.
datasetTrain = datasetTrain.apply(
    tf.data.experimental.assert_cardinality(len(generatorTrain))
)
datasetVal = datasetVal.apply(
    tf.data.experimental.assert_cardinality(len(generatorVal))
)

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
    callbacks_array.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                              patience=5, min_lr=0.001, cooldown=2))

    model.compile(loss=loss, optimizer=optimizer, metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5), 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives'])
elif args.model.startswith("CNN2Plus1D"):
    if args.model == "CNN2Plus1DLight":
        model = CNN2Plus1D_Light(seed=args.seed)
    elif args.model == "CNN2Plus1DFilters":
        model = CNN2Plus1D_Filters(seed=args.seed)
    elif args.model == "CNN2Plus1DLayers":
        model = CNN2Plus1D_Layers(seed=args.seed)
    else:
        model = CNN2Plus1D(seed=args.seed)

    loss = BinaryCrossentropy()  # Alternative for two label Classifications: Hinge Loss or Squared Hinge Loss
    lr = CosineDecay(initial_learning_rate=0.0, warmup_steps=n_batches_per_epoch * args.warmup, warmup_target=0.001, decay_steps=n_batches_per_epoch * (args.epochs - args.warmup), alpha=0.0)
    optimizer = AdamW(learning_rate=lr)
    lr_metric = helper_functions.get_lr_metric(optimizer)
    callbacks_array.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_acc', factor=0.1,
                              patience=10, min_lr=0.00001, cooldown=2))
    model.compile(loss=loss, optimizer=optimizer, metrics=[lr_metric, tf.keras.metrics.BinaryAccuracy(threshold=0.5), 'TrueNegatives', 'TruePositives', 'FalseNegatives', 'FalsePositives'])
elif args.model == 'MoViNets':
    # model = MoViNets()
    raise NotImplementedError
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
    log_dir=os.path.join(output_path, 'tensorboard_logs'),  # os.path.join(args.output_path, args.model, 'tensorboard_logs'),
    # don't think I need weight histogramms histogram_freq=1,
    update_freq='epoch'
))

callbacks_array.append(keras.callbacks.EarlyStopping(monitor='val_acc', patience=20))

callbacks_array.append(helper_functions.CSVLogger(filename=os.path.join(output_path, 'outputs_csv.log')))

tracker = OfflineEmissionsTracker(project_name="VVAD", experiment_id=args.model, country_iso_code="DEU", output_dir=output_path, output_file="codecarbon", tracking_mode="process") # gpu_ids=[0,1,2,3], on_csv_write="append/update"
tracker.start()

model.fit(x = datasetTrain,
                    epochs = args.epochs,
                    callbacks=callbacks_array,
                    validation_data = datasetVal,
                    use_multiprocessing=args.use_multiprocessing,
                    workers=args.workers,
                    max_queue_size=args.max_queue_size)

tracker.stop()
