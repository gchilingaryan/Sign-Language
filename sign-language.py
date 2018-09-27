import pandas as pd
import numpy as np
import os
from random import shuffle
from tqdm import *
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam

MODEL_NAME = "sign-language.h5"

def process_data():
    train_df = pd.read_csv('sign_mnist_train.csv')
    test_df = pd.read_csv('sign_mnist_test.csv')
    train_data = train_df.iloc[:,1:].values
    test_data = test_df.iloc[:,1:].values
    train_label = train_df['label'].values
    test_label = test_df['label'].values

    train_data = np.array(train_data).reshape((-1, 1, 28, 28)).astype(np.uint8) / 255.0
    test_data = np.array(test_data).reshape((-1, 1, 28, 28)).astype(np.uint8) / 255.0
    train_label = to_categorical(train_label, 25).astype(np.uint8)

    training_data = []
    for i, data in tqdm(enumerate(train_data)):
        label = train_label[i]
        training_data.append([np.array(data), np.array(label)])
    shuffle(training_data)

    testing_data = []
    for i, data in tqdm(enumerate(test_data)):
        testing_data.append([np.array(data), i+1])

    return training_data, testing_data, test_label

def train():
    training_data, testing_data, test_label = process_data()

    model = Sequential()
    model.add(Conv2D(32, 2, activation='relu', input_shape=(28,28,1)))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, 2, activation='relu'))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, 2, activation='relu'))
    model.add(Conv2D(64, 2, activation='relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(25, activation='softmax'))
    optimizer = Adam(lr=0.001)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    if os.path.exists(MODEL_NAME):
        model.load_weights(MODEL_NAME)
        print 'model exists'

    train = training_data[:-500]
    test = training_data[-500:]
    X = np.array([i[0] for i in train]).reshape([-1, 28, 28, 1])
    y = np.array([i[1] for i in train])

    test_x = np.array([i[0] for i in test]).reshape([-1, 28, 28, 1])
    test_y = np.array([i[1] for i in test])

    model.fit(X, y, epochs=5, verbose=1, validation_data=(test_x, test_y))

    model.save(MODEL_NAME)

    return model, testing_data, test_label

def test(model, testing_data, test_label):
    correct = 0

    for i, data in enumerate(testing_data):
        img_num = data[1]
        img_data = data[0]

        data = img_data.reshape(-1, 28, 28, 1)
        model_out = model.predict([data])[0]

        label = np.argmax(model_out)
        if test_label[i] == label:
            correct += 1

    print float(correct)/len(testing_data)


if __name__ == "__main__":
    model, testing_data, test_label = train()
    test(model, testing_data, test_label)
