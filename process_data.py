import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from tensorflow import keras



COLUMN_NAMES = ['MPG','Cylinders','Displacement','Horsepower','Weight',
                'Acceleration', 'Model Year', 'Origin']

def load_data(data_file, url):
    dataset_path = keras.utils.get_file(data_file, url)
    raw_dataset = pd.read_csv(dataset_path, names=COLUMN_NAMES,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)
    return raw_dataset


dataset = load_data("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
#default - 5 values at tail

#encapsulate pandas method in our wrapper method
def remove_na_values(dataframe):
    return dataframe.dropna()
    #dataset.pop(target_column)



def reformat_dataframe(dataframe, target_col, new_columns):
    for i in range(len(new_columns)):
        dataframe[new_columns[i]] = (target_col == (i+ 1)) * 1.0
    #return dataframe


def split_dataset(dataset, sample_frac, random_state = 0):
    train_dataset = dataset.sample(frac=sample_frac, random_state= random_state)
    test_dataset = dataset.drop(train_dataset.index)
    return (train_dataset, test_dataset)

def get_dataset_description(dataset):
    stats = dataset.describe()
    stats.pop("MPG")
    return stats.transpose()


def norm(x, mean, std):
  return (x - mean) / std


