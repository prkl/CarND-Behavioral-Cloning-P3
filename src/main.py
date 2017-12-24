import csv
import cv2
from os import listdir
import os.path as op
import numpy as np
import h5py
import tensorflow

# white_list = ['map_1_backward_1']
white_list = []
data_dir_name = 'my_data'
img_dir_name = 'IMG'
data_dir = '../' + data_dir_name
log_name = 'driving_log.csv'


def add_image(image, measurement, images, measurements):
    image_flip, measurement_flip = flip_image(image, measurement)

    images.append(image)
    measurements.append(measurement)
    images.append(image_flip)
    measurements.append(measurement_flip)

def flip_image(image, measurement):
    image_flipped = np.fliplr(image)
    measurement_flipped = -measurement
    return image_flipped, measurement_flipped

def get_image(path):
    path = image_paths_prefix[i] + '/' + img_dir_name + '/' + path.split(img_dir_name)[1]
    return cv2.imread(path)


lines = []
image_paths_prefix = []
subdirs = [op.join(data_dir, f) for f in listdir(data_dir) if op.isdir(op.join(data_dir, f))]
for dir in subdirs:
    if white_list and op.basename(dir) not in white_list:
        continue
    log_file = op.join(dir, log_name)
    if (not op.isfile(log_file)):
        continue
    with open(log_file) as log:
        reader = csv.reader(log)
        for line in reader:
            lines.append(line)
            image_paths_prefix.append(dir)

images = []
measurements = []
for i in range(len(lines)):
    line = lines[i]

    center_image = get_image(line[0])
    left_image = get_image(line[1])
    right_image = get_image(line[2])

    correction = 0.2
    center_steering = float(line[3])
    add_image(center_image, center_steering, images, measurements)
    add_image(left_image, center_steering + correction, images, measurements)
    add_image(right_image, center_steering - correction, images, measurements)



    # left_steering = center_steering + correction
    # right_steering = center_steering - correction
    #
    # images.append(center_image)
    # images.append(left_image)
    # images.append(right_image)
    # measurements.append(center_steering)
    # measurements.append(left_steering)
    # measurements.append(right_steering)
    #
    # center_image_flip, center_steering_flip = flip_image(center_image, center_steering)


    # image_flipped = np.fliplr(center_image)
    # measurement_flipped = -center_steering

X_train = np.array(images)
y_train = np.array(measurements)

from keras.models import Sequential
from keras.layers import Flatten, Dense, Convolution2D, MaxPooling2D, Cropping2D, Lambda

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(3, 160, 320)))

model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Convolution2D(6, 5, 5, activation='relu'))
# model.add(MaxPooling2D())
# model.add(Flatten())
# model.add(Dense(128))
# model.add(Dense(64))
# model.add(Dense(1))


# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=3, batch_size=128)

model.save('model.h5')
