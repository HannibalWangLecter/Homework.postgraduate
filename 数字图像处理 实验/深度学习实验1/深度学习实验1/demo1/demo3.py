from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt


#batch_size, num_classes and epochs
batch_size = 128
num_classes = 10
epochs = 10

# input image dimensions
img_rows, img_cols = 28, 28

# load minst and reshape the input data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
input_shape = (img_rows, img_cols, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.np_utils.to_categorical(y_train, num_classes)
y_test = keras.utils.np_utils.to_categorical(y_test, num_classes)

#construct a convolutional neuron network

######################################################################
model=Sequential()
model.add(Conv2D(8,(3,3),activation='relu',padding='same',input_shape=input_shape))
model.add(Conv2D(12,(3,3),activation='relu'))
model.add(Conv2D(14,(3,3),activation='relu'))
model.add(Conv2D(18,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(10,(3,3),activation='relu'))
model.add(Conv2D(12,(3,3),activation='relu'))
model.add(Conv2D(14,(3,3),activation='relu'))
model.add(Conv2D(18,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))#全连接层
model.add(Dense(num_classes, activation='softmax'))


#####################################################################

#model compile:loss function, optimiazer and metrics
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
#visualize the architecture of CNN
model.summary()

#train the model
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_test, y_test))
          
#save weights
model.save('mnist_demo3.h5')

#evaluate the model
score_test = model.evaluate(x_test, y_test, verbose=0)
score_train = model.evaluate(x_train,y_train,verbose=0)
print(score_test)
print(score_train)
