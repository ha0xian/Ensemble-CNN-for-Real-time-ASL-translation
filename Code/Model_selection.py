import os
from pathlib import Path
import tensorflow as tf
import Config
from Train_model import create_model, model_tuning
from Manage_csv import *
from Load_images import load_data


def train_all_models(models, numClass, inputShape, callback, trainData, valData,path, epochs=20,
                     ftepochs=50):
    """
    Function to train and evaluate all models, record results as csv and graphs,
     save results, models and weights into given directionary
    :param models: A dictionary to be updated
    :param numClass: number of class to be classified
    :param inputShape: dimensions of the input image
    :param callback: callback function
    :param trainData: training data
    :param valData:validation data
    :param path: a path to directionary to save csv
    :param epochs: number of epochs for training
    :param ftepochs: number of epoches in fine tuning
    :return: a dictionary with validation loss, accuracy, and path to saved model
    """
    modelPath = os.path.join(path, 'Models')
    weightPath = os.path.join(path, 'Weights')
    resultsPath = os.path.join(path, 'Results')
    Path(modelPath).mkdir(parents=True, exist_ok=True)
    Path(resultsPath).mkdir(parents=True, exist_ok=True)
    Path(weightPath).mkdir(parents=True, exist_ok=True)
    for k in range(8):
        modelNum = k + 1

        # Create current Model
        print("Current Model:")
        currModel, modelname = create_model(modelNum, numClass, 'imagenet', inputShape)
        currModel.fit(trainData, epochs=epochs, validation_data=valData, callbacks=[callback], steps_per_epoch=500)

        # Fine Tune Model
        currModel = model_tuning(currModel, fromLayer=0, inputShape=(128, 128, 3))
        currModel.fit(trainData, epochs=ftepochs, validation_data=valData, callbacks=[callback], steps_per_epoch=500)

        # Evaluate Model
        valLoss, valAcc = currModel.evaluate(valData)

        # Save model to dictionary and Path
        modelpathName = str(modelname + '_01.h5')
        currModelPath = os.path.join(modelPath, modelpathName)
        currModel.save(currModelPath)
        currModel.save_weights(os.path.join(weightPath, modelpathName))
        models = update_models(models, modelname, valLoss, valAcc, currModelPath)
        update_results(models, resultsPath)

    print('Trained All Models')
    return models

if __name__ == "__main__":
    callback = tf.keras.callbacks.EarlyStopping('val_accuracy', patience=5, restore_best_weights=True)
    trainData, valData = load_data(Config.trainDataPath, validation=True,validation_split=0.3, aug=True)
    resultPath = os.path.join(Config.resultsPath, 'All_training_models/')
    modelsList = {}
    train_all_models(modelsList, 29, (128, 128, 3), callback, trainData, valData,resultPath,10,5)