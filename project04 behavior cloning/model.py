import tensorflow as tf
# from keras import Sequential
from tensorflow.contrib.layers import flatten
from keras.preprocessing.image import ImageDataGenerator

from keras.models import Model
from keras.layers import Input, Convolution2D, Flatten, Dense, Dropout, ELU, Lambda, Cropping2D, Conv2D
from keras.callbacks import ModelCheckpoint, CSVLogger
import keras.backend as K

from os.path import join
import cv2
import numpy as np
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import random
import keras.backend as K
import os
import platform
import time

EPOCHS = 29 # 29
STEPS_PER_EPOCH = 2000  # epoch samples: STEPS_PER_EPOCH * BATCHSIZE
VALIDATION_STEPS = 100  # samples = VALIDATION_STEPS * BATCHSIZE
BATCHSIZE = 500

# Testing Now
EPOCHS = 3 # 29
STEPS_PER_EPOCH = 20  # epoch samples: STEPS_PER_EPOCH * BATCHSIZE
VALIDATION_STEPS = 1  # samples = VALIDATION_STEPS * BATCHSIZE
BATCHSIZE = 500

DEBUG = False
if (platform.system()== 'Darwin'):
    DEBUG = True

if DEBUG:
    EPOCHS = 1
    STEPS_PER_EPOCH = 5
    VALIDATION_STEPS = 1
    BATCHSIZE = 500


# print("\n--------------------------------")
# print(f"Epochs: {EPOCHS}")
# print(f"Samples per epoch: {STEPS_PER_EPOCH * BATCHSIZE}")
# print("--------------------------------\n")


DATA_DIR = './data'
TRAIN_VAL_SPLIT = 0.2


INPUT_CHANNELS = 3

CROP_HEIGHT = range(20, 140)

STEERING_CORRECTION = 0.3

NVIDIA_WIDTH = 66
NVIDIA_HEIGHT = 200

WIDTH = 320
HEIGHT = 160


def split_data():
    driving_log = DATA_DIR + '/driving_log.csv'
    with open(driving_log, 'r') as f:
        reader = csv.reader(f)
        driving_data = [row for row in reader][1:]

    train, validation = train_test_split(
        driving_data,
        test_size=TRAIN_VAL_SPLIT,
        random_state=1
    )

    return train, validation


def load_batch(data, augment):
    shuffled_data = shuffle(data)

    X = np.zeros(shape=(BATCHSIZE, 160, 320, 3), dtype=np.float32)
    y_steer = np.zeros(shape=(BATCHSIZE,), dtype=np.float32)

    loaded_elements = 0
    while loaded_elements < BATCHSIZE:
        ct_path, lt_path, rt_path, steer, throttle, brake, speed = shuffled_data.pop()
        steer = np.float32(steer)

        # Randomly a camera angle
        camera = random.choice(['frontal', 'left', 'right'])
        frame = cv2.imread(join(DATA_DIR, ct_path.strip()))
        if camera == 'frontal':
            steer = steer
        elif camera == 'left':
            # frame = preprocess(cv2.imread(join(DATA_DIR, lt_path.strip())))
            steer = steer + STEERING_CORRECTION
        elif camera == 'right':
            # frame = preprocess(cv2.imread(join(DATA_DIR, rt_path.strip())))
            steer = steer - STEERING_CORRECTION

        # Data Augmentation
        if augment:
            # Mirror image
            if random.choice([True, False]):
                frame = frame[:, ::-1, :]
                steer *= -1.

            # Add variation to steering
            steer += np.random.normal(loc=0, scale=0.1)

        X[loaded_elements] = frame
        y_steer[loaded_elements] = steer
        loaded_elements += 1

    return X, y_steer


def generate_batch(data, augment):
    while True:
        X, y_steer = load_batch(data, augment)
        yield X, y_steer


def get_model():
    input_frame = Input(shape=(HEIGHT, WIDTH, INPUT_CHANNELS))

    # Normalization
    x = Lambda(lambda x: x / 255.0 - 0.5)(input_frame)

    # Cropping
    x = Cropping2D(((70, 20), (0,0)))(x)

    # Convolution layers
    x = Convolution2D(24, 5, 5, border_mode='valid', subsample=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(48, 5, 5, border_mode='valid', subsample=(2, 2))(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid')(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)
    x = Convolution2D(64, 3, 3, border_mode='valid')(x)
    x = ELU()(x)
    x = Dropout(0.2)(x)

    # FC layers
    x = Flatten()(x)
    x = Dense(100)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(50)(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Dense(10)(x)
    x = ELU()(x)
    out = Dense(1)(x)

    # Model
    model = Model(input=input_frame, output=out)
    model.summary()
    model.compile(optimizer='adam', loss='mse')

    return model


if __name__ == '__main__':
    train_data, val_data = split_data()
    model = get_model()

    training_data_generator = generate_batch(train_data, True)
    validation_data_generator = generate_batch(val_data, False)

    model.fit_generator(
        # data
        generator=training_data_generator,
        validation_data=validation_data_generator,

        # params
        epochs=EPOCHS,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_steps=VALIDATION_STEPS,

        verbose=1
    )

    if not DEBUG:
        model.save('model.h5')


