from __future__ import absolute_import, division, print_function

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from process_data import *
from model import *

dataset = load_data("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

print(dataset.shape)

#sum of na values grouped by columns - 6 values of horsepower
print(dataset.isna().sum())
print(dataset.shape)

#remove na values
dataset = remove_na_values(dataset)


origin = dataset.pop('Origin')
reformat_dataframe(dataset, origin, ['USA','Europe','Japan'])

print(dataset.tail())

train_dataset, test_dataset = split_dataset(dataset, 0.8)
sb.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
plt.show()

train_stats = get_dataset_description(train_dataset)
print(train_stats)
#target column
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')


def norm(x, mean, std):
  return (x - train_stats['mean']) / train_stats['std']

#same mean and std for both train and test dataset
normed_train_data = norm(train_dataset, train_stats['mean'], train_stats['std'])
normed_test_data = norm(test_dataset, train_stats['mean'], train_stats['std'])

print(normed_train_data.tail())



model = build_model(train_dataset)
print(model.summary())

train_model
history = train_model(
    model, normed_train_data, train_labels, EPOCHS, 0.2)

print('')
print(history)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())




plot_history(hist, history.epoch,**{'xlabel':'Epoch',
                      'ylabel':'Mean Abs Error [MPG]',
                      'label1':'Train Error',
                      'label2':'Val Error',
                      'ylim':[0,5],
                      'ydata1': 'mean_absolute_error',
                      'ydata2': 'val_mean_absolute_error',
                      'plot_flag':False})

plot_history(hist, history.epoch, **{'xlabel':'Epoch',
                        'ylabel':'Mean Square Error [$MPG^2$]',
                      'label1':'Train Error',
                      'label2':'Val Error',
                      'ylim':[0,20],
                      'ydata1':'mean_squared_error',
                    'ydata2':'val_mean_squared_error',
                      'plot_flag':True})

model = build_model(train_dataset)

# patience - number of epochs to wait for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = train_model(
    model, normed_train_data, train_labels, EPOCHS, 0.2, callback_list=[early_stop, PrintDash()])



loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print('')
print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

plot_prediction(test_labels, test_predictions)

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch

plot_history(hist, history.epoch, **{'xlabel':'Epoch',
                      'ylabel':'Mean Abs Error [MPG]',
                      'label1':'Train Error',
                      'label2':'Val Error',
                      'ylim':[0,5],
                      'ydata1': 'mean_absolute_error',
                      'ydata2': 'val_mean_absolute_error',
                      'plot_flag':False})

plot_history(hist, history.epoch, **{'xlabel':'Epoch',
                        'ylabel':'Mean Square Error [$MPG^2$]',
                      'label1':'Train Error',
                      'label2':'Val Error',
                      'ylim':[0,20],
                      'ydata1':'mean_squared_error',
                    'ydata2':'val_mean_squared_error',
                      'plot_flag':True})


plot_error_histogram(test_predictions, test_labels)




