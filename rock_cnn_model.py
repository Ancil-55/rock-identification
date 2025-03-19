# -*- coding: utf-8 -*-

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

model = Sequential()
model.add(Convolution2D(32,(3,3), input_shape=(64,64,3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(units=128, init='uniform', activation='relu'))
model.add(Dense(units=128, init='uniform', activation='relu'))
model.add(Dense(units=128, init='uniform', activation='relu'))
model.add(Dense(units=128, init='uniform', activation='relu'))
model.add(Dense(units=13, init="uniform", activation="softmax"))

from keras.preprocessing.image import ImageDataGenerator
train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r'F:\Projects\Rock Datasets\dataset\dataset\train',target_size=(64,64),batch_size=32,class_mode='categorical')
x_test=test_datagen.flow_from_directory(r'F:\Projects\Rock Datasets\dataset\dataset\test',target_size=(64,64),batch_size=32,class_mode='categorical')

x_train.class_indices

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=["accuracy"])
model.fit_generator(x_train, steps_per_epoch=500, epochs=25, validation_data=x_test, validation_steps=50)

model.save("trained_model.h5")

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

model = load_model("trained_model.h5")

img = image.load_img(r"F:\Sample\g21.jpg",target_size=(64,64))

x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)

pred = model.predict_classes(x)
pred


