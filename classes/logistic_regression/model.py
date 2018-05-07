#!/usr/bin/python
# This code was developed using https://medium.com/@the1ju/simple-logistic-regression-using-keras-249e0cc9a970
# Build the model of a logistic classifier

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
nb_epoch = 20
input_dim = 784

# the data, shuffled and split between train and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, input_dim)
X_test = X_test.reshape(10000, input_dim)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = build_logistic_model(input_dim, nb_classes)

model.summary()

# compile the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)

print('Test score:', score[0])
print('Test accuracy:', score[1])
