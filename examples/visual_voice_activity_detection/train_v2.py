import numpy as np
import argparse
from tensorflow.python.data import Dataset
import tensorflow as tf
keras = tf.keras
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
import matplotlib.pyplot as plt

from paz.datasets import VVAD_LRS3
from paz.models.classification import CNN2Plus1D

parser = argparse.ArgumentParser(description='Paz VVAD Training')
parser.add_argument('-p', '--data_path', type=str,
                    default='.keras/paz/datasets',
                    help='Path from your home dir to dataset directory')
parser.add_argument('-m', '--model', type=str,
                    default='CNN2Plus1D',
                    help='Model you want to train',
                    choices=['VVADLRS3', 'CNN2Plus1D', 'MoViNets', 'ViViT'])
parser.add_argument('-b', '--batch_size', type=int,
                    default=16,
                    help='Batch size for training and validation')

args = parser.parse_args()

generatorTrain = VVAD_LRS3(split="train", path=args.data_path)
generatorVal = VVAD_LRS3(split="val", path=args.data_path)

datasetTrain = Dataset.from_generator(generatorTrain, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))
datasetVal = Dataset.from_generator(generatorVal, output_signature=(tf.TensorSpec(shape=(38, 96, 96, 3)), tf.TensorSpec(shape=(), dtype=tf.int8)))

datasetTrain = datasetTrain.batch(args.batch_size)
datasetVal = datasetVal.batch(args.batch_size)

# Load model defined in args.model
model = None
if args.model == 'VVADLRS3':
    # model = VVAD_LRS3()
    raise NotImplementedError
elif args.model == 'CNN2Plus1D':
    model = CNN2Plus1D()
elif args.model == 'MoViNets':
    # model = MoViNets()
    raise NotImplementedError
elif args.model == 'ViViT':
    # model = ViViT()
    raise NotImplementedError

loss = BinaryCrossentropy(from_logits=True)  # Alternative for two label Classifications: Hinge Loss or Squared Hinge Loss
optimizer = Adam(learning_rate=0.0001)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

history = model.fit(x=datasetTrain,
                    epochs=2,
                    validation_data=datasetVal)


# def plot_history(history):
#   """
#     Plotting training and validation learning curves.
#
#     Args:
#       history: model history with all the metric measures
#   """
#   fig, (ax1, ax2) = plt.subplots(2)
#
#   fig.set_size_inches(18.5, 10.5)
#
#   # Plot loss
#   ax1.set_title('Loss')
#   ax1.plot(history.history['loss'], label = 'train')
#   ax1.plot(history.history['val_loss'], label = 'test')
#   ax1.set_ylabel('Loss')
#
#   # Determine upper bound of y-axis
#   max_loss = max(history.history['loss'] + history.history['val_loss'])
#
#   ax1.set_ylim([0, np.ceil(max_loss)])
#   ax1.set_xlabel('Epoch')
#   ax1.legend(['Train', 'Validation'])
#
#   # Plot accuracy
#   ax2.set_title('Accuracy')
#   ax2.plot(history.history['accuracy'],  label = 'train')
#   ax2.plot(history.history['val_accuracy'], label = 'test')
#   ax2.set_ylabel('Accuracy')
#   ax2.set_ylim([0, 1])
#   ax2.set_xlabel('Epoch')
#   ax2.legend(['Train', 'Validation'])
#
#   plt.show()
#
# plot_history(history)