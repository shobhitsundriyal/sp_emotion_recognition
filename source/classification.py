import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import normalize
#import utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Dense, Input, Dropout, Flatten, Conv2D, BatchNormalization, Activation, MaxPooling2D
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import plot_model

from tensorflow.keras.utils import to_categorical

#Hyperparams and global variables
img_size = 48
batch_size = 32
num_classes = 3
epochs = 30
r_mapping = {0:'Fear', 1:'Happy', 2:'Sad'}
mapping = {'Fear':0, 'Happy':1,'Sad':2}
datagen = ImageDataGenerator(rotation_range=20, horizontal_flip=True, validation_split=0.15, 
                             width_shift_range=0.1, height_shift_range=0.1)

#Model
model = Sequential()

model.add(Conv2D(64, (3,3), padding='same', input_shape=(img_size, img_size, 1)))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (5,5), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(512, (3,3), padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(512))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(3, activation='softmax'))

opt = Adam(lr=0.005)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy',])
model.summary()
#---------

'''
The following dummy code for demonstration.
'''


def train_a_model(trainfile):
    '''
    :param trainfile:
    :return:
    '''
    data = pd.read_csv(trainfile)
    X, y = data.iloc[:, 1:].values, data['emotion'].map(mapping).values
    X_norm = X/255.
    X = X_norm.reshape(X.shape[0], 48, 48, 1)
    y = to_categorical(y, num_classes=3)
    datagen.fit(X)

    history = model.fit(datagen.flow(X, y, batch_size=32),
                        steps_per_epoch=len(X) / 32, epochs=epochs)



def test_the_model(testfile):
    '''

    :param testfile:
    :return:  a list of predicted values in same order of
    '''
    test_df = pd.read_csv(testfile)
    test = test_df.values.astype('float32') / 255.
    test = test.reshape(test.shape[0], 48, 48, 1)
    test_op = model.predict_classes(test)
    #print(test_op)
    ret = []
    for label in test_op:
        clas = r_mapping[label]
        ret.append(clas)
    return ret