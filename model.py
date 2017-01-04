import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.misc import imread, imresize

from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Dense, Flatten, Dropout, MaxPooling2D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam


def resize_image(image_path):
    path = image_path.replace(' ', '')
    img = imread(path)
    resized_img = imresize(img, (32, 32))
    return resized_img


def a_model():
    model = Sequential([
        BatchNormalization(input_shape=(32, 32, 3)),
        Conv2D(32, 3, 3, activation='relu', border_mode='same'),
        Conv2D(32, 3, 3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(64, 3, 3, activation='relu', border_mode='same'),
        Conv2D(64, 3, 3, activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1)
    ])
    return model

# load the data
driving_log = pd.read_csv('driving_log.csv')
image_paths = pd.concat([driving_log['center'], driving_log['left'], driving_log['right']])
image_paths = np.array(image_paths, dtype=pd.Series)
mirror_paths = driving_log['center']
mirror_paths = np.array(mirror_paths, dtype=pd.Series)
angles = pd.concat([driving_log['steering'], driving_log['steering'] + 0.25, driving_log['steering'] - 0.25,
                   -driving_log['steering']])
angles = np.array(angles, dtype=pd.Series)

# preprocess images
images = [resize_image(path) for path in image_paths]
images.extend([np.fliplr(resize_image(path)) for path in mirror_paths])
images = np.array(images)
angles = np.array([np.asarray([angle], np.float32) for angle in angles])

images_training, images_validation, angles_training, angles_validation = train_test_split(images, angles, test_size=0.2,
                                                                                          random_state=42)
nb_training = images_training.shape[0]
nb_validation = images_validation.shape[0]
nb_epoch = 18

#  train the model
my_model = a_model()
my_model.summary()
my_model.compile(optimizer=Adam(lr=0.0001), loss='mse')
generator = ImageDataGenerator(width_shift_range=0.2, fill_mode='nearest')

# only save the best weights
checkpoint = ModelCheckpoint('model.h5', monitor='val_loss', verbose=1, save_best_only=True)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1)
callbacks = [early_stop, checkpoint]

my_model.fit_generator(generator.flow(images_training, angles_training, batch_size=128), nb_training, nb_epoch,
                       validation_data=generator.flow(images_validation, angles_validation, batch_size=128),
                       callbacks=callbacks, nb_val_samples=nb_validation)

# save the model
with open("model.json", "w") as json_file:
    json_file.write(my_model.to_json())
