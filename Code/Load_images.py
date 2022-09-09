import math
import os
from random import shuffle

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from Preprocess_img import preprocess_img
from tensorflow.keras.utils import to_categorical

def load_cv2(path, imgW=256, imgH=256, num=1):
    # Load images and labels with OpenCV into numpy array
    # Images are save in different folder where the folders are the label for the image
    X, Y = [], []
    for dir1 in os.listdir(path): # Loop through each folder
        imgList = []
        imgLen = len(os.listdir(os.path.join(path, dir1)))
        i=1
        for img in os.listdir(os.path.join(path, dir1)):
            imgList.append(img)
        shuffle(imgList) # Shuffle the image
        for img in imgList:
            if i <= math.ceil(num*imgLen): # Load partial of the data due to memory issues
                imgPath = os.path.join(path, dir1, img)
                image = preprocess_img(imgPath, imgW, imgH)
                X.append(image)
                Y.append(dir1)
                i+=1
#         print('{0} images Found in class {1}'.format(i, str(dir1)))
    Y, catY, labelMap = convert_labels(Y)
    print('Found {0} images in {1} classes'.format(len(X), len(np.unique(Y))))
    return np.array(X), np.array(Y), catY, labelMap

def load_data(path,validation=False,size=(128,128,3),validation_split=0,aug=True):
    """
    Function to load images and label with keras & augment data
    :param path: path to input directory
    :param validation: Boolean to split into train and validation
    :param size: Load the image with specified size
    :param validation_split: validation split
    :param aug: Boolean if images should be augmented with ImageDataGenerator
    :return: Tensors of Batch images with labels
    """
    # Load images and label with keras & augment data
    if aug: # Augment data with ImageDataGenrator
        datagen = ImageDataGenerator(
                            horizontal_flip=True,
                            rotation_range=15,
                            brightness_range=[0.2,1.3],
                            width_shift_range=0.1,
                            height_shift_range=0.1,
                            validation_split=validation_split,
                            fill_mode='reflect')
        noAug = ImageDataGenerator(validation_split=validation_split)
    else:
        datagen = ImageDataGenerator(validation_split=validation_split)
    if validation:
        traindata= datagen.flow_from_directory(path,
                                               target_size=size,
                                               subset='training',
                                               seed=123,
                                               batch_size=32,
                                               shuffle=False)
        valdata = noAug.flow_from_directory(path,
                                            target_size=size,
                                            subset='validation',
                                            seed=123,
                                            batch_size=32,
                                            shuffle=False)
        return traindata, valdata
    elif validation==False:
        traindata=datagen.flow_from_directory(path,
                                              target_size=size,
                                              batch_size=32)
        return traindata

def convert_labels(labels):
    # Functions to convert labels into dictionary with kets and values
    labelMap = {label: num for num, label in enumerate(np.unique(labels))}
    for i in range(len(labels)):
        labels[i] = labelMap[labels[i]]
    return labels,to_categorical(labels), labelMap

def reverse_map(labelMap):
    return {v: k for k, v in labelMap.items()}


def split_train_val_test(X, Y, test_size=0.2, val_size=0.2, validation=True):
    # function to split data into train, validation and test
    if validation:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size)
        Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=val_size)
        return Xtrain, Xval, Xtest, Ytrain, Yval, Ytest
    elif validation==False:
        Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=test_size)
        return Xtrain, Xtest, Ytrain, Ytest

