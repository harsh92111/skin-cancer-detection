# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

classifier = Sequential()

classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())

classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'C:/skincancerdataset/train',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'C:/skincancerdataset/test',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=2637,
        epochs=3,
        validation_data=test_set,
        validation_steps=660)

#classifier.save('1cnn.model')

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('C:/Users/harsh/Pictures/m1.jpg', target_size=(64, 64))
test_image = image.img_to_array(test_image)
test_image =np.expand_dims(test_image, axis=0)
result=classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction='malignant'
else:
    prediction='benign'

print(prediction)