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

# the data, shuffled and split between train and test sets
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
#if you change the architecture, you should change the following code
#####################################################################
model = Sequential()
model.add(Conv2D(10,(3,3),activation='relu',input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

####################################################################


#model compile:loss function, optimiazer and metrics
model.compile(loss=keras.metrics.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
              
#visualize the architecture of CNN
model.summary()          

#load the well-trained parameters, including weights and bias
model.load_weights('mnist_demo1.h5')

#obtain the model prediction in testing set
model_prediction = model.predict(x_test)

#try to find the differencees between  model_prediction and y_test to find the miss judgement
###################################################################
plt.figure(figsize=(20, 4))  #画一张画布
b = [2,0,1,8,2,1,0,6,1,1,0,7]  #学号
n = len(b)  #学号长度
for j in range(len(b)):  #对学号进行循环	
    for i in range(10000):  #对验证集的样本进行循环		
        pos_model = np.argmax(model_prediction[i,:])#真实标签
        pos_y = np.argmax(y_test[i,:])#模型输出		
        if pos_model != pos_y:  #如果判错			
                if pos_y == b[j]: #如果判错且正好等于学号						
                    ax = plt.subplot(1, n, j + 1)				
                    plt.imshow(x_test[i].reshape(28, 28))					
                    plt.gray()					
                    break

plt.show()








#################################################################
				
				
				
				

