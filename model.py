import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense, Dropout
from skimage.io import imread
import numpy as np

# load the data
driving_log = pd.read_csv('./driving_log.csv')
#image_paths = pd.concat([driving_log['center'], driving_log['left'], driving_log['right']])
#angles = pd.concat([driving_log['steering'], driving_log['steering'] + 0.3, driving_log['steering'] - 0.3])
image_paths = driving_log['center']
angles = driving_log['steering']
paths_training, paths_validation, angles_training, angles_validation = train_test_split(image_paths, angles,
                                                                                        test_size=0.15, random_state=42)
nb_training = paths_training.shape[0]
nb_validation = paths_validation.shape[0]
nb_epoch = 2


# build the model
def a_model():
    model = Sequential()

    model.add(Convolution2D(64, 11, 11, border_mode='valid', subsample=(4, 4), input_shape=(160, 320, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(256, 5, 5, border_mode='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.add(Convolution2D(384, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(384, 3, 3))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(256, 1, 1))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(4096))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Dense(1))

    model.summary()
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

    return model


def generator(ins, outs):
    while 1:
        for (image_path, angle) in zip(ins, outs):
            path = ('./' + image_path).replace(' ', '')
            img = imread(path)[None, :, :, :]
            ang = np.asarray([angle], np.float64)
            yield img, ang

# train the model
my_model = a_model()
my_model.fit_generator(generator(paths_training, angles_training), nb_training, nb_epoch,
                       validation_data=generator(paths_validation, angles_validation), nb_val_samples=nb_validation)

# save the model
with open("model.json", "w") as json_file:
    json_file.write(my_model.to_json())
my_model.save_weights('model.h5')
