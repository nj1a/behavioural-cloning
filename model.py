import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imresize

from keras.models import Sequential
from keras.layers import Lambda, Conv2D, ELU, Dense, Flatten
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


def process_image(image_path, flip=False):
    path = image_path.replace(' ', '')
    img = imread(path)
    # fit into input shape (66, 200, 3)
    resized = imresize(img, (100, 200))
    cropped = resized[34:, :, :]
    if flip:
        return np.fliplr(cropped)
    return cropped


def a_model():
    model = Sequential([
        Lambda(lambda x: x / 127.5 - 1., input_shape=(66, 200, 3)),
        Conv2D(24, 5, 5, init='he_normal', subsample=(2, 2)),
        ELU(),
        Conv2D(36, 5, 5, init='he_normal', subsample=(2, 2)),
        ELU(),
        Conv2D(48, 5, 5, init='he_normal', subsample=(2, 2)),
        ELU(),
        Conv2D(64, 3, 3, init='he_normal', subsample=(1, 1)),
        ELU(),
        Conv2D(64, 3, 3, init='he_normal', subsample=(1, 1)),
        Flatten(),
        ELU(),
        Dense(1164, init='he_normal'),
        ELU(),
        Dense(100, init='he_normal'),
        ELU(),
        Dense(50, init='he_normal'),
        ELU(),
        Dense(10, init='he_normal'),
        ELU(),
        Dense(1, init='he_normal')
    ])
    return model

# load the data
driving_log = pd.read_csv('driving_log.csv')
image_paths = pd.concat([driving_log['center'], driving_log['left'], driving_log['right'], driving_log['center']])
image_paths = np.array(image_paths, dtype=pd.Series)
angles = pd.concat([driving_log['steering'], driving_log['steering'] + 0.25, driving_log['steering'] - 0.25],
                   driving_log['steering'] * -1)
angles = np.array(angles, dtype=pd.Series)

# preprocess images
images = np.array([process_image(path) for path in image_paths])
angles = np.array([np.asarray([angle], np.float64) for angle in angles])

images_training, images_validation, angles_training, angles_validation = train_test_split(images, angles, test_size=0.2,
                                                                                          random_state=4242)
nb_training = images_training.shape[0]
nb_validation = images_validation.shape[0]
nb_epoch = 8

my_model = a_model()
my_model.summary()
my_model.compile(optimizer=Adam(lr=0.0001), loss='mse')
generator = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, channel_shift_range=0.1)
my_model.fit_generator(generator.flow(images_training, angles_training, batch_size=128), nb_training, nb_epoch,
                       validation_data=generator.flow(images_validation, angles_validation, batch_size=128),
                       nb_val_samples=nb_validation)

# save the model
with open("model.json", "w") as json_file:
    json_file.write(my_model.to_json())
my_model.save_weights('model.h5')
