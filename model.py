import csv
import cv2
import numpy as np

# import basic scikit learn method for processing data
from sklearn.model_selection import train_test_split
from random import shuffle
import sklearn
import math

# import all necessary keras modules necessary
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers import Lambda, Cropping2D
from math import ceil


# A subroutine code basically returns a batch of data that will be augmented and shuffled 
# for training and validation fit by keras
def generator(samples, batch_size):
    num_samples = len(samples)
    correction_factor = 0.4
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2RGB)
                images.append(center_image)

                left_name = './data/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.cvtColor(cv2.imread(left_name), cv2.COLOR_BGR2RGB)
                images.append(left_image)

                right_name = './data/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.cvtColor(cv2.imread(right_name), cv2.COLOR_BGR2RGB)
                images.append(right_image)

                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                # add correction factor for left and right camera images
                angles.append(center_angle + correction_factor)
                angles.append(center_angle - correction_factor)

                # Flipping the images to artificially create additional data
                images.append(cv2.flip(center_image, 1))
                angles.append(center_angle * -1.0)

                images.append(cv2.flip(left_image, 1))
                # reverse angle by multiplying -1 to the angle
                angles.append(center_angle * -1.0 + correction_factor)
                images.append(cv2.flip(right_image, 1))
                angles.append(center_angle * -1.0 - correction_factor)           
            
            # convert the data into numpy arrays
            X = np.array(images)
            y = np.array(angles)
            # yield the prediçter images and predictor angles and shuffle them
            yield sklearn.utils.shuffle(X, y)


samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

# Set our batch size
batch_size = 32

# create a test train split for 20% of data as test data
(train_data, validation_data) = train_test_split(samples[1:], test_size=0.2)

# create generator objects representing train data and validation data
train_generator = generator(train_data, batch_size=batch_size)
validation_generator = generator(validation_data, batch_size=batch_size)

# Using Nvidianet architecture
model = Sequential()
# normalize the pixel values
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
# crop the images to only scope to the area of concern
model.add(Cropping2D(cropping=((60, 25), (0, 0))))
# convolusions using 24 filters created using 5x5 kernels - by default using 'valid' padding and strides of 2 pixels in all directions
model.add(Conv2D(24, (5, 5), activation='relu', strides=(2, 2)))
# convolusions using 36 filters
model.add(Conv2D(36, (5, 5), activation='relu', strides=(2, 2)))
# convolusions using 48 filters
model.add(Conv2D(48, (3, 3), activation='relu', strides=(2, 2)))
# convolusions using 64 filters
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
# convolusions using 64 filters
model.add(Conv2D(64, (3, 3), activation='relu', strides=(2, 2)))
# drop some of the weights to reduce overfitting
model.add(Dropout(0.3))
# flatten the resulting image to individual pixes as one input
model.add(Flatten())
# pass it through fully connected network of 125 layers
model.add(Dense(125, activation='relu'))
# drop 30% of weights connected to next layer at random during training
model.add(Dropout(0.3))
# pass it through fully connected network of 50 layers
model.add(Dense(50, activation='relu'))
# drop 30% of weights at ramdom
model.add(Dropout(0.3))
# pass it through fully connected network of 10 layers
model.add(Dense(10, activation='relu'))
# drop 30% of weights at ramdom
model.add(Dropout(0.3))
model.add(Dense(1))

# using adam optimizer insted of SGD with learning rate
model.compile(loss='mse', optimizer='adam')
# train the above network for 10 iterations, passing the validation and training data generators for trianing
model.fit_generator(
    train_generator,
    steps_per_epoch=ceil(len(train_data) / batch_size),
    validation_data=validation_generator,
    validation_steps=ceil(len(validation_data) / batch_size),
    epochs=10,
    verbose=1,
    )
# save the model
model.save('model.h5')