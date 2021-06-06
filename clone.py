import csv
import cv2
import numpy as np
import sklearn.model_selection as model_selection
from sklearn.model_selection import train_test_split
from random import shuffle
import sklearn
import math

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)                
                images.append(center_image)
                
                left_name = './data/IMG/'+batch_sample[1].split('/')[-1]
                left_image = cv2.imread(left_name)                
                images.append(left_image)
                
                right_name = './data/IMG/'+batch_sample[2].split('/')[-1]
                right_image = cv2.imread(right_name)                
                images.append(right_image)
                
                center_angle = float(batch_sample[3])
                angles.append(center_angle)
                angles.append(center_angle)
                angles.append(center_angle)
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


samples = []
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

            
# Set our batch size
batch_size=32

train_data, validation_data = train_test_split(samples[1:], test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_data, batch_size=batch_size)
validation_generator = generator(validation_data, batch_size=batch_size)



from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Lambda, Cropping2D
from keras import backend as K
from math import ceil
from keras.layers.convolutional import Convolution2D

row, col, ch = 160, 320, 3 # Trimmed image format
# model = Sequential()


model = Sequential()
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((50,20), (0,0))))
model.add(Convolution2D(24,5,5,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(36,5,5,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(48,3,3,border_mode='valid'))
model.add(MaxPooling2D((2,2)))
model.add(Activation('relu'))
model.add(Convolution2D(64,3,3,border_mode='valid'))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(1164))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(100))  
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Dropout(0.2))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Dropout(0.2))
model.add(Activation('linear'))
model.add(Dense(1))

# compile and fit the model
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=ceil(len(train_data)/batch_size), \
                    validation_data=validation_generator, \
                    validation_steps=ceil(len(validation_data)/batch_size), \
                    epochs=5, verbose=1)
model.save('model.h5')
exit()
    