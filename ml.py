import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import joblib

def regression_tree(train_X, val_X, train_y, val_y):

    rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=60, random_state=1, verbose=2)

    choice = int(input("1 to train, 2 to load: "))
    file = 'data/' + input("Filename to load or dump: ")

    if choice == 1:
        rf_model.fit(train_X, train_y)
        joblib.dump(rf_model, file)
        rf_val_predictions = rf_model.predict(val_X)

    elif choice == 2:
        loaded_rf = joblib.load(file)
        rf_val_predictions = loaded_rf.predict(val_X)

    else:
        for i in range(50, 150, 30):
            rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=15, random_state=1, verbose=2)
            rf_model.fit(train_X, train_y)
            joblib.dump(rf_model, file)
            rf_val_predictions = rf_model.predict(val_X)
            print(str(i) + ': ' + str(mean_absolute_error(rf_val_predictions, val_y)))

    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)

    print("Validation MAE for Random Forest Model: {:,.5f}".format(rf_val_mae))


train = pd.read_csv('C:/mnist_train.csv')
val = pd.read_csv('C:/mnist_test.csv')

train_cols = list(train.columns)
train.rename(columns={train_cols[i]:str(i) for i in range(len(train_cols))}, inplace=True)

val_cols = list(val.columns)
val.rename(columns={val_cols[i]:str(i) for i in range(len(val_cols))}, inplace=True)

train_X = train[train.columns[1:len(train.columns)]] / 255.0
val_X = val[val.columns[1:len(val.columns)]] / 255.0

train_y = train['0']
val_y = val['0']

# regression_tree(train_X, val_X, train_y, val_y)

train_X_array = np.array(train_X)

for i in range(784):
    print(f"{train_X_array[49035][i]:.0f}", end='')
    if (i + 1) % 28 == 0 and i != 0:
        print('')

