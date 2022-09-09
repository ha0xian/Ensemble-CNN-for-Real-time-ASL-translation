import tensorflow as tf
import keras_tuner as kt
import Config
from functools import partial
from tensorflow import keras
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from Load_images import load_data

'''
Hyperparameters tuning for ensemble model 
'''

def create_hp_model(modelList, units, lr, dropoutRate, hpSecLayer, units2):
    """
    Function to build ensemble model with given hyperparameters
    :param modelList: List of models to form ensemble model
    :param units: number of neurons in the 1st dense layer for meta model
    :param lr: learning rate
    :param dropoutRate: dropout rate
    :param hpSecLayer: Boolean, to add second dense layer
    :param units2: number of neurons in the 2nd dense layer for meta model
    :return: a compiled model
    """
    # Avoid ensemble model's layers having the same name
    for m in range(len(modelList)):
        model = modelList[m]
        for l in model.layers:
            l._name = l.name + '_' + str(m + 1)
            l.trainable = False

    # Create Ensemble model
    modelsInput = keras.layers.Input(shape=(128, 128, 3))
    modelsOutput = [model(modelsInput) for model in modelList]
    ensembleInput = tf.keras.layers.concatenate(modelsOutput)
    hidden = Dense(units, activation='relu')(ensembleInput)
    if hpSecLayer:
        hidden = Dense(units2, activation='relu')(hidden)
    dropOut = Dropout(dropoutRate)(hidden)
    ensembleOutput = Dense(29, activation='softmax')(dropOut)

    model = Model(inputs=modelsInput, outputs=ensembleOutput)
    model.compile(optimizer=Adam(learning_rate=lr), loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def create_build_model(modelList, hp):
    # Define Hyperparameters to tuned
    hpUnits = hp.Choice("units", [128, 256, 512, 1024])
    hpSecLayer = hp.Boolean("secondLayer")
    hpUnits2 = hp.Choice("units2", [128, 256, 512, 1024])
    hpLr = hp.Float("lr", min_value=0.0001, max_value=0.01, sampling='log')
    hpDropoutRate = hp.Choice("dropoutRate", [0.0, 0.2, 0.3, 0.35, 0.4])

    model = create_hp_model(modelList, hpUnits, hpLr, hpDropoutRate, hpSecLayer, hpUnits2)

    return model

def create_best_model(tuner):
    # Extract the best model from the tuner
    bestHps = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(bestHps)
    print("Number of Units1: {0} \nNumber of Units2: {1}\nLearning Rate: {2}\nSecondLayer: {3} \nDropout Rate:{4}"
          .format(bestHps.get("units"),bestHps.get("units2"),bestHps.get("lr"),str(bestHps.get("secondLayer")),bestHps.get("dropoutRate")))
    return model
if __name__ == "__main__":

    _,valData = load_data(Config.segmentTrainPath,validation=True,validation_split=0.3)
    testData = load_data(Config.segmentTestPath2,validation=False)

    # Form a list of base models
    model1 = load_model(Config.vgg19Path)
    model2 = load_model(Config.resnet50Path)
    model3 = load_model(Config.mobilenetPath)
    modelList = [model1,model2,model3]

    # Build models and tuner for hyperparameters tuning
    build_model = partial(create_build_model,modelList)
    tunerDL = kt.Hyperband(build_model, 'val_accuracy', 10,factor=2,hyperband_iterations=2)

    # Starts hyperparameter tuning
    tunerDL.search(valData, validation_data=testData, batch_size=16)
    model = create_best_model(tunerDL)

    print(tunerDL.results_summary(1))

