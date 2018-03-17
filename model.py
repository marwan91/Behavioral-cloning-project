import os
import csv

samples = []
# Loading the main dataset
with open('data3/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
    samples.remove(samples[0])
print('the length of the main sample is ',len(samples))
    
samples2 = []
#Loading the supplementary (corrective) dataset
with open('data5/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples2.append(line)
    samples2.remove(samples2[0])

from sklearn.model_selection import train_test_split
#Splitting the main dataset into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.2)
#Adding the corrective dataset to the training set
train_samples=train_samples+samples2


import cv2
import numpy as np
import sklearn

from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D, Cropping2D
from keras.layers.pooling import MaxPooling2D


def mygenerator(samples, batch_size=32):
    num_samples = len(samples)
    shuffle(samples)
    images = []
    angles = []
    for offset in range(1, num_samples, batch_size):
        batch_samples = samples[offset:offset+batch_size]

        for batch_sample in batch_samples:
            
            #reading the image paths in each line and reformatting the strings into python paths
            name = batch_sample[0]
            name = name[33:]
            name = name.replace('\\','/')
                
            name_left = batch_sample[1]
            name_left = name_left[33:]
            name_left = name_left.replace('\\','/')
                
            name_right = batch_sample[2]
            name_right = name_right[33:]
            name_right = name_right.replace('\\','/')
            
            #Loading the images from the dataset folder and converting their color space to RGB
            
            center_image = cv2.cvtColor(cv2.imread(name),cv2.COLOR_BGR2RGB)  
            left_image = cv2.cvtColor(cv2.imread(name_left),cv2.COLOR_BGR2RGB) 
            right_image = cv2.cvtColor(cv2.imread(name_right),cv2.COLOR_BGR2RGB) 
            
            #Reading the steering angle corresponding to the images
            center_angle = float(batch_sample[3])
            
            # Multiplying the steering angle by a factor was shown to make the model more responsive during testing 
            center_angle=center_angle*3
            
            # Augmenting the dataset by flipping the images and their corresponing steering angles    
            flipped_center_image=cv2.flip(center_image,1)
            
            flipped_center_angle=(-1)*center_angle
            flipped_right_angle=(-1)*center_angle
            flipped_left_angle=(-1)*center_angle
            
            flipped_left_image=cv2.flip(left_image,1)
            flipped_right_image=cv2.flip(right_image,1)
            
            images.append(flipped_center_image)
            angles.append(flipped_center_angle)
            
            images.append(flipped_left_image)
            angles.append(flipped_left_angle)
            
            images.append(flipped_right_image)
            angles.append(flipped_right_angle)
                
            images.append(center_image)
            angles.append(center_angle)
            images.append(left_image)
            angles.append(center_angle)
            images.append(right_image)
            angles.append(center_angle)
                
                

        X_train = np.array(images)
        y_train = np.array(angles)
            
    return sklearn.utils.shuffle(X_train, y_train)

# compile and train the model using the mygenerator function
train_x , train_y = mygenerator(train_samples, batch_size=32)
valid_x , valid_y = mygenerator(validation_samples, batch_size=32)

# Model Architecture
model = Sequential()
model.add(Cropping2D(cropping=((50,20), (0,0)),data_format='channels_last', input_shape=(160,320,3)))
# Preprocess incoming data, centered around zero with small standard deviation 
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Conv2D(12,(5,5),strides=(2, 2), padding='valid',activation='relu'))
model.add(Conv2D(24,(5,5),strides=(2, 2), padding='valid',activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dropout (0.5, noise_shape=None, seed=None))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')

model.fit(train_x,train_y, batch_size=100, epochs=7, verbose=1, callbacks=None, validation_split=0.0, validation_data=(valid_x,valid_y), shuffle=True, initial_epoch=0, steps_per_epoch=None)


model.save('model.h5')

print('model saved')
