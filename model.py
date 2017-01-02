import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imresize

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, ELU, Dense, Flatten, Activation, Dropout, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


def process_image(image_path):
    path = image_path.replace(' ', '')
    img = imread(path)
    # fit into input shape (66, 200, 3)
    # resized = imresize(img, (100, 200))
    # cropped = resized[34:, :, :]
    cropped = imresize(img, (32, 32))
    return cropped


def a_model():
    # model = Sequential([
    #     Lambda(lambda x: x / 127.5 - 1., input_shape=(66, 200, 3)),
    #     Conv2D(24, 5, 5, init='he_normal', subsample=(2, 2)),
    #     ELU(),
    #     Conv2D(36, 5, 5, init='he_normal', subsample=(2, 2)),
    #     ELU(),
    #     Conv2D(48, 5, 5, init='he_normal', subsample=(2, 2)),
    #     ELU(),
    #     Conv2D(64, 3, 3, init='he_normal', subsample=(1, 1)),
    #     ELU(),
    #     Conv2D(64, 3, 3, init='he_normal', subsample=(1, 1)),
    #     Flatten(),
    #     ELU(),
    #     Dense(1164, init='he_normal'),
    #     ELU(),
    #     Dense(100, init='he_normal'),
    #     ELU(),
    #     Dense(50, init='he_normal'),
    #     ELU(),
    #     Dense(10, init='he_normal'),
    #     ELU(),
    #     Dense(1, init='he_normal')
    # ])
    model = Sequential()
    model.add(Conv2D(32, 3, 3, border_mode='same', input_shape=(32, 32, 3,)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    return model

# load the data
driving_log = pd.read_csv('driving_log.csv')
image_paths = pd.concat([driving_log['center'], driving_log['left'], driving_log['right']])
image_paths = np.array(image_paths, dtype=pd.Series)
mirror_paths = driving_log['center']
mirror_paths = np.array(mirror_paths, dtype=pd.Series)
angles = pd.concat([driving_log['steering'], driving_log['steering'] + 0.3, driving_log['steering'] - 0.25,
                   -driving_log['steering']])
angles = np.array(angles, dtype=pd.Series)

# preprocess images
images = [process_image(path) for path in image_paths]            
images.extend([np.fliplr(process_image(path)) for path in mirror_paths])
images = np.array(images)
angles = np.array([np.asarray([angle], np.float64) for angle in angles])

images_training, images_validation, angles_training, angles_validation = train_test_split(images, angles, test_size=0.2,
                                                                                          random_state=4242)
nb_training = images_training.shape[0]
nb_validation = images_validation.shape[0]
nb_epoch = 8

my_model = a_model()
my_model.summary()
my_model.compile(optimizer=Adam(lr=0.0001), loss='mse')
generator = ImageDataGenerator(width_shift_range=0.2, fill_mode='nearest')
my_model.fit_generator(generator.flow(images_training, angles_training, batch_size=128), nb_training, nb_epoch,
                       validation_data=generator.flow(images_validation, angles_validation, batch_size=128),
                       nb_val_samples=nb_validation)

# save the model
with open("model.json", "w") as json_file:
    json_file.write(my_model.to_json())
my_model.save_weights('model.h5')
