import os
import random
import Config
import cv2
import numpy as np
import tensorflow as tf
from keras.layers import concatenate, Dropout, Conv2DTranspose
from sklearn.model_selection import train_test_split
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D
from tensorflow.keras.models import Model


def UNET(input_shape=(128, 128, 3), last_activation='sigmoid'):
    """
    Constructing U-Net model. via https://github.com/jakeret/tf_unet
    by Akeret, Joel and Chang, Chihway and Lucchi, Aurelien and Refregier, Alexandre
    :param input_shape: input shape in Input layer
    :param last_activation: activation function used in the Output layer
    :return: a compiled U-Net model
    """
    inputs = tf.keras.layers.Input(shape=input_shape)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    d1 = Dropout(0.1)(conv1)
    conv2 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d1)
    b = BatchNormalization()(conv2)

    pool1 = MaxPooling2D(pool_size=(2, 2))(b)
    conv3 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    d2 = Dropout(0.2)(conv3)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d2)
    b1 = BatchNormalization()(conv4)

    pool2 = MaxPooling2D(pool_size=(2, 2))(b1)
    conv5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    d3 = Dropout(0.3)(conv5)
    conv6 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d3)
    b2 = BatchNormalization()(conv6)

    pool3 = MaxPooling2D(pool_size=(2, 2))(b2)
    conv7 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    d4 = Dropout(0.4)(conv7)
    conv8 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d4)
    b3 = BatchNormalization()(conv8)

    pool4 = MaxPooling2D(pool_size=(2, 2))(b3)
    conv9 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    d5 = Dropout(0.5)(conv9)
    conv10 = Conv2D(512, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d5)
    b4 = BatchNormalization()(conv10)

    conv11 = Conv2DTranspose(512, (4, 4), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer='he_normal')(b4)
    x = concatenate([conv11, conv8])
    conv12 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x)
    d6 = Dropout(0.4)(conv12)
    conv13 = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d6)
    b5 = BatchNormalization()(conv13)

    conv14 = Conv2DTranspose(256, (4, 4), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer='he_normal')(b5)
    x1 = concatenate([conv14, conv6])
    conv15 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x1)
    d7 = Dropout(0.3)(conv15)
    conv16 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(d7)
    b6 = BatchNormalization()(conv16)

    conv17 = Conv2DTranspose(128, (4, 4), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer='he_normal')(b6)
    x2 = concatenate([conv17, conv4])
    conv18 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x2)
    d8 = Dropout(0.2)(conv18)
    conv19 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d8)
    b7 = BatchNormalization()(conv19)

    conv20 = Conv2DTranspose(64, (4, 4), activation='relu', padding='same', strides=(2, 2),
                             kernel_initializer='he_normal')(b7)
    x3 = concatenate([conv20, conv2])
    conv21 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(x3)
    d9 = Dropout(0.1)(conv21)
    conv22 = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(d9)

    outputs = Conv2D(1, (1, 1), activation=last_activation, padding='same', kernel_initializer='he_normal')(conv22)
    model2 = Model(inputs=inputs, outputs=outputs)
    model2.compile(loss='mean_squared_error', optimizer=optimizers.RMSprop(learning_rate=0.0001))
    return model2

def load_data(imgPath, maskPath, size=(128,128), valSplit=0):
    """
    Function to load RGB images as X, and Binary images as Y
    :param imgPath: directory of input images
    :param maskPath: directory of output images
    :param size: size of the images
    :param valSplit: validation split between 0-1
    :return: numpy array of training and testing X and Y images
    """
    handList = os.listdir(imgPath)
    maskList = os.listdir(maskPath)
    handList.sort()
    maskList.sort()
    X,Y= [], []
    for img in range(len(handList)):
        handString = handList[img].split('.')[0]
        maskString = maskList[img].split('.')[0]
        if handString == maskString:
            currImg = cv2.imread(os.path.join(imgPath,handList[img]))
            currImg = cv2.cvtColor(currImg, cv2.COLOR_BGR2RGB)
            currImg = cv2.resize(currImg,size)
            currImg = currImg.astype('float32') / 255
            X.append(currImg)

            currMask= cv2.imread(os.path.join(maskPath, maskList[img]))
            currMask= cv2.resize(currMask, size)
            currMask = currMask.astype('float32')/255
            Y.append(currMask)
    if valSplit == 0:
        print('Found {0} images'.format(len(X)))
        return np.array(X), np.array(Y)
    else:
        Xtrain, Xval, Ytrain, Yval = train_test_split(X, Y, test_size=valSplit, random_state=42)
        print('Found {0} images for training, {1} for validation'.format(len(Xtrain), len(Xval)))
        return Xtrain, Xval, Ytrain, Yval

if __name__ == "__main__":
    img_folder_1 = Config.dataSegment1
    mask_folder_1 = Config.dataMask1

    img_folder_2 = Config.dataSegment2
    mask_folder_2 = Config.dataMask2

    img_folder_3 = Config.dataSegment3
    mask_folder_3 = Config.dataMask3

    Xtrain1,Xval1,Ytrain1,Yval1 = load_data(img_folder_1, mask_folder_1,(128,128), 0.3)
    Xtrain2,Xval2,Ytrain2,Yval2 = load_data(img_folder_2, mask_folder_2,(128,128),0.2)
    Xtrain3,Xval3,Ytrain3,Yval3 = load_data(img_folder_3, mask_folder_3,(128,128),0.3)
    Xtrain= np.concatenate((Xtrain1,Xtrain2,Xtrain3))
    Ytrain = np.concatenate((Ytrain1,Ytrain2,Ytrain3))

    # Concatenate all of the data
    Xval = np.concatenate((Xval1,Xval2,Xval3))
    Yval = np.concatenate((Yval1, Yval2,Yval3))

    del Yval1, Yval2, Ytrain1, Ytrain2, Xval1, Xval2, Xtrain1, Xtrain2,Xtrain3,Xval3,Ytrain3,Yval3

    # Shuffle 3 of the datasets
    random.seed(10)
    random.shuffle(Xtrain)
    random.seed(10)
    random.shuffle(Ytrain)
    random.seed(20)
    random.shuffle(Xval)
    random.seed(20)
    random.shuffle(Yval)

    unet = UNET()
    unet.fit(Xtrain, Ytrain, epochs=10, validation_data=(Xval,Yval), batch_size=32)
    unet.save(Config.UNetPath)
