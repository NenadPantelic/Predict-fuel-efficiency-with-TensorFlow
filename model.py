import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

EPOCHS = 1000

def build_model(train_dataset):
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='mean_squared_error',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model


class PrintDash(keras.callbacks.Callback):

  counter = 0
  dashes = ['/', '-','\\','-']
  def on_epoch_end(self, epoch, logs):
    PrintDash.counter = (PrintDash.counter + 1) % len(PrintDash.dashes)
    if epoch % 100 == 0: print('')
    print(PrintDash.dashes[PrintDash.counter], end='')


def train_model(model, data, labels, epochs, val_split, callback_list = [PrintDash()]):

    return model.fit(
    data, labels,
    epochs=epochs, validation_split = val_split, verbose=0,
    callbacks=callback_list)


def plot_prediction(test_labels, test_predictions):
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_history(hist, epoch, **kwargs):
    hist['epoch'] = epoch
    plt.figure()

    plt.xlabel(kwargs['xlabel'])
    plt.ylabel(kwargs['ylabel'])
    plt.plot(hist['epoch'], hist[kwargs['ydata1']],
             label=kwargs['label1'])
    plt.plot(hist['epoch'], hist[kwargs['ydata2']],
             label=kwargs['label2'])
    plt.ylim(kwargs['ylim'])
    plt.legend()
    if(kwargs['plot_flag']):
        plt.show()



def plot_error_histogram(test_predictions, test_labels, bins_num = 25):
    error = test_predictions - test_labels
    plt.hist(error, bins=bins_num)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()



