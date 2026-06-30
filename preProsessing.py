import pandas as pd
import numpy as np

def load_data(path):
    df = pd.read_csv("data/house_price.csv")
    return df

def train_test_split(df, test_size = 0.2):
    shuffled = df.sample(frac=1, random_state=2)
    split = int(len(df)*(1 - test_size))

    train = shuffled[:split]
    test = shuffled[split:]
    return train, test

def feature_train_split(df):
    X = df.iloc[:, : -1].values
    Y = df.iloc[:, -1].values
    return X, Y

def stardardize(X):
    mean = np.mean(X, axis = 0)
    std = np.std(X, axis = 0)
    X = (X - mean) / std
    return X, mean, std