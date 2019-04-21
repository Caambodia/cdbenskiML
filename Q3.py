'''
PS#2
Q3 Inception Module for CIFAR-10 dataset

'''
import keras
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Dropout
from keras.layers import Input
from keras.utils import np_utils
from keras.datasets import cifar10

epochs = 20

# Get the data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Get the data ready
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Create imput
input_img = Input(shape = (32, 32, 3))


# Create Volumes for the Inception module 1
volume_1 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)

volume_2 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
volume_2a = Conv2D(64, (3,1), padding='same', activation='relu')(volume_2)
volume_2b = Conv2D(64, (1,3), padding='same', activation='relu')(volume_2)
volume_2a = Dropout(0.5)(volume_2a)
volume_2b = Dropout(0.5)(volume_2b)

volume_3 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
volume_3 = Conv2D(64, (3,3), padding='same', activation='relu')(volume_3)
volume_3a = Conv2D(64, (3,1), padding='same', activation='relu')(volume_3)
volume_3b = Conv2D(64, (1,3), padding='same', activation='relu')(volume_3)
volume_3a = Dropout(0.5)(volume_3a)
volume_3b = Dropout(0.5)(volume_3b)

volume_4 = Conv2D(64, (1,1), padding='same', activation='relu')(input_img)
volume_4 = Conv2D(64, (3,3), padding='same', activation='relu')(volume_4)
volume_4 = Conv2D(64, (5,5), padding='same', activation='relu')(volume_4)
volume_4a = Conv2D(64, (3,1), padding='same', activation='relu')(volume_4)
volume_4b = Conv2D(64, (1,3), padding='same', activation='relu')(volume_4)
volume_4a = Dropout(0.5)(volume_4a)
volume_4b = Dropout(0.5)(volume_4b)

volume_5 = MaxPooling2D((3,3), strides=(1,1), padding='same')(input_img)
volume_5 = Conv2D(64, (1,1), padding='same', activation='relu')(volume_5)
volume_5 = Dropout(0.5)(volume_5)

# Concatenate all volumes of the Inception module
inception_module = keras.layers.concatenate([volume_1, volume_2a, volume_2b, volume_3a, volume_3b, volume_5], axis = 3)


out = Flatten()(inception_module)
out = Dense(10, activation='softmax')(out)


model = Model(inputs = input_img, outputs = out)
print(model.summary())

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=512)


scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


