import os
from pathlib import Path
import Config
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import InceptionV3, VGG19, ResNet50, Xception, MobileNet, DenseNet201, MobileNetV2, \
    VGG16, ResNet152
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Average, Maximum
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from Manage_csv import update_models, update_results
from Load_images import load_data


def build_base_model(modelNum, numClasses, weights=None, inputShape=(256, 256, 3)):
    """
    Function to create different pre-trained model
    :param modelNum: int 1-9 to create different pre-trained models;
                    1-InceptionV3; 2-VGG19; 3-Xception; 4-MobileNetV2; 5-ResNet50;
                    6-DenseNet201; 7-MobileNet; 8-VGG16; 9-ResNet152
    :param numClasses: number of classification class
    :param weights: 'None', 'imagenet' or given .h5 file
    :param inputShape: dimensions of input data
    :return: model and name of the model
    """
    kwargs = {'weights': weights,
              'include_top': False,
              'input_shape': inputShape,
              'classes': numClasses
              }
    if modelNum == 1:
        model = InceptionV3(**kwargs)
        print('InceptionV3 model is created')
        modelName = 'InceptionV3'
    elif modelNum == 2:
        model = VGG19(**kwargs)
        print('VGG19 model is created')
        modelName = 'VGG19'
    elif modelNum == 3:
        model = Xception(**kwargs)
        print('Xception model is created')
        modelName = 'Xception'
    elif modelNum == 4:
        model = MobileNetV2(**kwargs)
        print('MobileNetV2 model is created')
        modelName = 'MobileNetV2'
    elif modelNum == 5:
        model = ResNet50(**kwargs)
        print('ResNet50 model is created')
        modelName = 'ResNet50'
    elif modelNum == 6:
        model = DenseNet201(**kwargs)
        print('DenseNet201 model is created')
        modelName = 'DenseNet201'
    elif modelNum == 7:
        model = MobileNet(**kwargs)
        print('MobileNet model is created')
        modelName = 'MobileNet'
    elif modelNum == 8:
        model = VGG16(**kwargs)
        print('VGG16 model is created')
        modelName = 'VGG16'
    elif modelNum == 9:
        model = ResNet152(**kwargs)
        print('ResNet152 model is created')
        modelName = 'ResNet152'
    else:
        print('Model not found, model = None')
        model = None
    return model, modelName


def create_model(modelNum, numClasses, weights=None, inputShape=(256, 256, 3), trainable=False,trainingRate=0.001):
    """
    Function to create pre-trained model from build_base_model(), modified model for fine-tuning and compile it
    :param modelNum: int 1-9 to create different pre-trained models;
                    1-InceptionV3; 2-VGG19; 3-Xception; 4-MobileNetV2; 5-ResNet50;
                    6-DenseNet201; 7-MobileNet; 8-VGG16; 9-ResNet152
    :param numClasses: number of classification task
    :param weights: 'None', 'imagenet' or given .h5 file
    :param inputShape: dimensions of the input data
    :param trainable: Boolean; True, unfreeze the layers; False, freeze the layer
    :param trainingRate: training rate for Adam optimizer
    :return: compiled model, and model name
    """
    kwargs = {'modelNum': modelNum,
              'weights': weights,
              'inputShape': inputShape,
              'numClasses': numClasses
              }

    base_model, modelName = build_base_model(**kwargs)
    if base_model == None:
        return None

    base_model.trainable = trainable
    x = base_model.output
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.2)(x)
    predictions = Dense(numClasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(learning_rate=trainingRate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model, modelName

def ensemble_avg_model(modelList, inputShape=(128,128,3), trainingRate=0.001):
    """
    Function to form an ensemble model, with average voting classifier, based on given list of model
    :param modelList: List of models to form ensemble model
    :param inputShape: dimensions of the input data
    :param trainingRate: learning rate for Adam optimizer
    :return: a compiled ensemble model with average voting classifier
    """
    for m in range(len(modelList)):
        model = modelList[m]
        for l in model.layers:
            l._name = l.name + '_'+str(m+1)
    modelInput = keras.layers.Input(shape=inputShape)
    modelOutput = [model(modelInput) for model in modelList] # Concatenate output of each model
    ensembleOutput = Average()(modelOutput) # Compute the average value
    ensembleModel = Model(modelInput, ensembleOutput)
    ensembleModel.compile(optimizer=Adam(learning_rate = trainingRate), loss='categorical_crossentropy', metrics=['accuracy'])
    return ensembleModel

def ensemble_denseLayer_model(modelList, numClasses=29, inputShape=(128,128,3)):
    """
     Function to form an ensemble model, with dense layer as meta-model, based on given list of model
    :param modelList: List of models to form ensemble model
    :param numClasses:number of classification classes
    :param inputShape: dimensions of the input data
    :return: a compiled ensemble model
    """
    for m in range(len(modelList)):
        model = modelList[m]
        for l in model.layers:
            l._name = l.name + '_'+str(m+1)
            l.trainable = False
    modelsInput = keras.layers.Input(shape=inputShape)
    modelsOutput = [model(modelsInput) for model in modelList] # Concatenate output of each model
    ensembleInput = tf.keras.layers.concatenate(modelsOutput)
    hidden = Dense(512, activation='relu')(ensembleInput) # Append 1st dense layer
    hidden = Dense(512, activation='relu')(hidden) # Append 2nd dense layer
    dropOut = Dropout(0.3)(hidden) # Dropout to avoid overfitting
    ensembleOutput = Dense(numClasses, activation='softmax')(dropOut) # Output layer
    model = Model(inputs=modelsInput, outputs=ensembleOutput)
    model.compile(optimizer=Adam(learning_rate =0.0002), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def ensemble_maximum_model(modelList, inputShape=(128,128,3), trainingRate=0.001):
    """
    Function to form an ensemble model, with maximum voting classifier, based on given list of model
    :param modelList: List of models to form ensemble model
    :param inputShape: dimensions of the input data
    :param trainingRate: learning rate for Adam optimizer
    :return: a compiled ensemble model with maximum voting classifier
    """
    modelInput = keras.layers.Input(shape=inputShape)
    modelOutput = [model(modelInput) for model in modelList] # Concatenate the output of the each model
    ensembleOutput = Maximum()(modelOutput) # Compute the maximum value
    ensembleModel = Model(modelInput, ensembleOutput)
    ensembleModel.compile(optimizer=Adam(learning_rate = trainingRate), loss='categorical_crossentropy', metrics=['accuracy'])
    return ensembleModel

def model_tuning(model,fromLayer=0,trainingRate=0.00001,printing=False):
    """
    Function to unfreeze model's layer for fine-tuning
    :param model: classification model
    :param fromLayer: an int to specify which layer to start unfreezing
    :param trainingRate: learning rate for Adam optimizer while compiling
    :param printing: Boolean to print model summary
    :return: a compiled model
    """
    outputModel = model
    for layer in outputModel.layers[-fromLayer:]:
        if isinstance(layer, BatchNormalization):
            layer.trainable=False
        else:
            layer.trainable=True
    if printing:
        for layer in outputModel.layers:
            print("{}: {}".format(layer, layer.trainable))

    outputModel.compile(optimizer=Adam(learning_rate = trainingRate), loss='categorical_crossentropy', metrics=['accuracy'])
    return outputModel


def get_final_model(m1,m2,m3,weightPath, modeltype):
    # function to get final ensemble model with loaded .h5 weights
    ml = [m1,m2,m3]
    if modeltype == 'Avg':
        model = ensemble_avg_model(ml)
    if modeltype == 'Max':
        model = ensemble_maximum_model(ml)
    if modeltype == 'DL':
        model = ensemble_denseLayer_model(ml)
    model.load_weights(weightPath)
    return model

if __name__ == "__main__":
    callback = tf.keras.callbacks.EarlyStopping('val_accuracy',patience=5, restore_best_weights=True)
    ftcallback = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=2, restore_best_weights=True)
    traindataPath = Config.segmentTrainPath
    trainData, valData = load_data(traindataPath, validation=True,validation_split=0.3, aug=True)

    # Create base models
    vgg19,_ = create_model(2,29,'imagenet',(128,128,3))
    resnet,_ = create_model(5,29,'imagenet',(128,128,3))
    mobilenet,_ = create_model(7,29,'imagenet',(128,128,3))

    # Starts training base models
    vgg19.fit(trainData, validation_data=valData, epochs=20, callbacks=callback)
    resnet.fit(trainData, validation_data=valData, epochs=20, callbacks=callback)
    mobilenet.fit(trainData, validation_data=valData, epochs=20, callbacks=callback)

    # Unfreeze base models' layers
    vgg19 = model_tuning(vgg19)
    resnet = model_tuning(resnet)
    mobilenet = model_tuning(mobilenet)

    # Fine-tuning
    vgg19.fit(trainData, validation_data=valData, epochs=10, callbacks=ftcallback)
    resnet.fit(trainData, validation_data=valData, epochs=10, callbacks=ftcallback)
    mobilenet.fit(trainData, validation_data=valData, epochs=10, callbacks=ftcallback)

    # Save the model
    vgg19.save(Config.vgg19Path)
    resnet.save(Config.resnet50Path)
    mobilenet.save(Config.mobilenetPath)

    # a list of base models for ensemble model
    modelList = [vgg19,resnet,mobilenet]

    # Create ensemble model
    enAvg = ensemble_avg_model(modelList)
    enMax = ensemble_maximum_model(modelList)
    enDL = ensemble_denseLayer_model(modelList, 29)

    # Train Meta model
    enDL.fit(valData, epochs=8)

    # Save ensemble model weights
    enAvg.save_weights(Config.enAvgW)
    enMax.save_weights(Config.enMaxW)
    enDL.save_weights(Config.enDLW)

