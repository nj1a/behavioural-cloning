import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Flatten, Dense, Dropout
from scipy.misc import imread, imresize
import numpy as np

# load the data
driving_log = pd.read_csv('./driving_log.csv')
# image_paths = pd.concat([driving_log['center'], driving_log['left'], driving_log['right']])
# angles = pd.concat([driving_log['steering'], driving_log['steering'] + 0.3, driving_log['steering'] - 0.3])
image_paths = driving_log['center']
angles = driving_log['steering']
paths_training, paths_validation, angles_training, angles_validation = train_test_split(image_paths, angles,
                                                                                        test_size=0.15, random_state=42)
nb_training = paths_training.shape[0]
nb_validation = paths_validation.shape[0]
nb_epoch = 5


def a_model():
    model = Sequential()
    model.add(BatchNormalization(input_shape=(66, 200, 3)))
    model.add(Convolution2D(24, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(36, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(48, 5, 5, activation='relu', subsample=(2, 2)))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', subsample=(1, 1)))
    model.add(Flatten())
    model.add(Dense(1164, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(50, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Dense(1, activation='relu'))

    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    return model


def generator(ins, outs):
    while 1:
        for (image_path, angle) in zip(ins, outs):
            path = ('./' + image_path).replace(' ', '')
            img = imread(path)
            # fit into input shape (66, 200, 3)
            resized = imresize(img, (100, 200))
            cropped = resized[None, 34:, :, :]
            ang = np.asarray([angle], np.float64)
            yield cropped, ang

my_model = a_model()
my_model.fit_generator(generator(paths_training, angles_training), nb_training, nb_epoch,
                       validation_data=generator(paths_validation, angles_validation), nb_val_samples=nb_validation)

# save the model
with open("model.json", "w") as json_file:
    json_file.write(my_model.to_json())
my_model.save_weights('model.h5')
