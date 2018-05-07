
# CIFAR 10 Data

# MODIFIED: Anup Sethuram for "IISC-CCE Deep Learning course - Sriram Ganapthy"
# Build the model for a convolution neural network

from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras.datasets import cifar10

from keras import backend as K
from keras.optimizers import SGD

import time
import numpy

start_time = time.time()


batch_size = 32
epochs = 15

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# Input image dimensions
img_rows, img_cols = 32, 32
img_size = 1
img_depth = 3
channels = 1

# load data from CIFAR-10
(x_train_orig, y_train_orig), (x_validation, y_validation) = cifar10.load_data()

print ("Original CIFAR data shape:", x_train_orig.shape)
print ("Original CIFAR data shape:", y_train_orig.shape)

x_train = x_train_orig[:42000]
x_test = x_train_orig[42000:]
y_train = y_train_orig[:42000]
y_test = y_train_orig[42000:]

print (x_train.shape)
print (x_test.shape)
print (y_train.shape)
print (y_test.shape)

num_classes = 10

# if K.image_dim_ordering() == 'th':
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols, img_depth)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols, img_depth)
    x_validation = x_validation.reshape(x_validation.shape[0], channels, img_rows, img_cols, img_depth)
    input_shape = (channels, img_rows, img_cols, img_depth)
    print ("DEBUG: Setting channels first ..")
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, channels)
    x_train = x_validation.reshape(x_validation.shape[0], img_rows, img_cols, img_depth, channels)
    input_shape = (img_rows, img_cols, img_depth, channels)
    print ("DEBUG: Setting channels last ..")

# if K.image_data_format() == 'channels_first':
#     x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols, img_depth)
#     x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols, img_depth)
#     input_shape = (1, img_rows, img_cols, img_depth)
# else:
#     x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, 1)
#     x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, 1)
#     input_shape = (img_rows, img_cols, img_depth, 1)


x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)
y_validation = keras.utils.to_categorical(y_validation)

# Model
model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))


# momentum = 0.8
# sgd = SGD(momentum=momentum)
sgd = SGD(lr=0.1)


model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

# Using the validation data in the final testing.
score = model.evaluate(x_validation, y_validation, verbose=0)
print('Test loss:', score[0]*100)
print('Test accuracy:', score[1]*100)

print("--- %s seconds ---" % (time.time() - start_time))
