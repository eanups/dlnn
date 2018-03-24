#!/usr/bin/python
# This code was developed using https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970
# MODIFIED: Anup Sethuram for "IISC-CCE Deep Learning course - Sriram Ganapthy"
# Build the model of a logistic classifier

import time

from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.datasets import mnist
from keras.utils import np_utils


def build_logistic_model(input_dimension, output_dim):
    l_model = Sequential()
    l_model.add(Dense(output_dim, input_dim=input_dimension, activation='softmax'))

    return l_model

batch_size = 128
nb_classes = 10
nb_epoch = 15
input_dim = 784

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print 'Initial Training Data Size : ', X_train.size
print "Initial Training Data shape: ", X_train.shape
print "y train shape ", y_train.shape
print "y test shape ", y_test.shape


X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim).astype('float32')

X_training = X_train[0:50000].astype('float32')
y_training = y_train[0:50000]

X_validation = X_train[50000:60000].astype('float32')
y_validation = y_train[50000:60000]


print " Training Data shape: ", X_training.shape
print " Validation Data shape: ", X_validation.shape

print "y training shape ", y_training.shape
print "y validation shape ", y_validation.shape


X_training /= 255
X_validation /= 255
X_test /= 255

print(X_training.shape[0], 'train samples')
print(X_validation.shape[0], 'validation samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_tr = np_utils.to_categorical(y_training, nb_classes)
y_val = np_utils.to_categorical(y_validation, nb_classes)
y_te = np_utils.to_categorical(y_test, nb_classes)

model = build_logistic_model(input_dim, nb_classes)

model.summary()

# compile the model
start_time = time.time()
learning_rate = 0.001
decay_rate = learning_rate / nb_epoch
momentum = 0.8
sgd = SGD(lr=learning_rate, momentum=momentum, decay=decay_rate, nesterov=False)

model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_training, y_tr,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_validation, y_val))

score = model.evaluate(X_test, y_te, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
