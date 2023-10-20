import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.utils import to_categorical
from keras.models import load_model

import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

import joblib


def get_high_index(values):

    values_nums = []

    for j in range(len(values)):

        max_value = values[j][0]
        max_index = 0

        for i in range(1, len(values[j])):
            if values[j][i] > max_value:
                max_value = values[j][i]
                max_index = i

        values_nums.append(max_index)

    return values_nums


def cnn(train_X, val_X, train_y, val_y):

    train_X_array = np.array(train_X)
    val_X_array = np.array(val_X)

    train_y_array = np.array(train_y)
    val_y_array = np.array(val_y)

    train_X_array = train_X_array.reshape(59999, 28, 28, 1)
    val_X_array = val_X_array.reshape(9999, 28, 28, 1)

    train_y_array = to_categorical(train_y_array)
    val_y_array = to_categorical(val_y_array)

    choice = int(input("1 to train, 2 to load: "))
    file = "data/" + input("Filename to load or dump: ") + '.keras'

    if choice == 1:

        # create model
        model = Sequential()

        # add model layers
        model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28, 28, 1)))
        model.add(Conv2D(32, kernel_size=3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        # compile model using accuracy to measure model performance
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # model.summary()
        # model.get_weights()

        model.fit(train_X_array, train_y_array, validation_data=(val_X_array, val_y_array), epochs=3, verbose=1)

        open(file, 'w').close()
        model.save(file)

        cnn_predictions = model.predict(val_X_array)
        cnn_mae = mean_absolute_error(cnn_predictions, val_y_array)

        print("Validation MAE for CNN Model: {:,.5f}".format(cnn_mae))

    elif choice == 2:
        index = 30

        ld_model = load_model(file)

        four_d = val_X_array[index:index + 2][:][:]

        # print(four_d.shape)
        # three_d = val_X_array[index][:][:].reshape(28, 28, 1)
        # plt.imshow(three_d)
        # plt.show()
        # print(ld_model.predict(four_d))

        print(get_high_index(ld_model.predict(four_d)))


def regression_tree(train_X, val_X, train_y, val_y):

    rf_model = RandomForestRegressor(n_jobs=-1, n_estimators=60, random_state=1, verbose=2)

    choice = int(input("1 to train, 2 to load: "))
    file = 'data/' + input("Filename to load or dump: ") + ".txt"

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
train.rename(columns={train_cols[i]: str(i) for i in range(len(train_cols))}, inplace=True)

val_cols = list(val.columns)
val.rename(columns={val_cols[i]: str(i) for i in range(len(val_cols))}, inplace=True)

t_X = train[train.columns[1:len(train.columns)]] / 255.0
v_X = val[val.columns[1:len(val.columns)]] / 255.0

t_y = train['0']
v_y = val['0']

# regression_tree(t_X, v_X, t_y, v_y)

cnn(t_X, v_X, t_y, v_y)
