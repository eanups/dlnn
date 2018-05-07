from __future__ import print_function
import keras

from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D
from keras import backend as K
import time

batch_size = 256
num_classes = 10
epochs = 15

# input image dimensions
img_rows, img_cols, img_depth, channels = 32, 32, 3, 1

VALIDATION_SPLIT = 8 / 50

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols, img_depth)
    x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols, img_depth)
    input_shape = (channels, img_rows, img_cols, img_depth)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depth, channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depth, channels)
    input_shape = (img_rows, img_cols, img_depth, channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3),
                 activation='relu',
                 input_shape=input_shape))

model.add(MaxPooling3D(pool_size=(2, 2, 1)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

start_time = time.time()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(lr=0.05),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_split=VALIDATION_SPLIT
          )
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0] * 100)
print('Test accuracy:', score[1] * 100)

print("--- %s seconds ---" % (time.time() - start_time))
