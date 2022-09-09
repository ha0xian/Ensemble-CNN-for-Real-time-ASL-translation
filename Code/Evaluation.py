import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import Config
import os
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.models import load_model

from Train_model import get_final_model
from Manage_csv import update_results
from Load_images import load_cv2


def get_img_text(score, labels):
    return labels[np.argmax(score)]

def get_scores(Y, X,model,className=None,display=False,output_dict=False):
    """
    Functions to evaluate models with classification report, accuracy score and confusion matrix
    :param Y: input Label
    :param X: input images used for prediction
    :param model: model to evaluate
    :param className: Labels' name
    :param display: Boolean to print results
    :param output_dict: Boolean to convert classfication report into dictionary (to transport to csv)
    :return: accuracy score, confusion matrix, classification report
    """
    yPred = model.predict(X)

    if className is None:
        className=np.unique(Y)

    if len(yPred.shape)==2:
        yPred=np.argmax(yPred,axis=1) # get predicted X class

    accScore = accuracy_score(Y, yPred)
    confusionMatrix = confusion_matrix(Y, yPred)
    report = classification_report(Y, yPred,target_names=className, output_dict=output_dict)
    if display:
        print('Accuracy Score: ', accScore)
        print('Classification Report: \n', report)
    return accScore, confusionMatrix, report

def plot_confusion_matrix(cf, classNames,savePath=None, title='Confusion matrix', show=True):
    # Plot confusion matrix with Sea born, and save it to savePath
    plt.figure(figsize = (17,12))
    plt.ylabel('True Label')
    plt.xlabel('Predicted label')
    plt.xticks(fontsize=12,rotation=45)
    plt.yticks(fontsize=12)
    sns.heatmap(cf, annot=True, xticklabels = classNames, yticklabels =classNames, cbar=False, cmap="Reds")
    plt.title(title, fontsize = 23)
    plt.tight_layout()
    cff = plt.gcf()
    if show:
        plt.imshow()

    if savePath !=None:
        cff.savefig(savePath,bbox_inches='tight', dpi=300)

def test_all_models(models, path, testData):
    for k in models:
        print(k)
        models[k]['val_loss'], models[k]['val_acc'] = models[k]['model'].evaluate(testData)
    update_results(models, path)

if __name__ == "__main__":
    # Load images and labels
    tx1, ty1, _, labelMap = load_cv2(Config.segmentTestPath1,128,128)
    tx2, ty2, _, _ = load_cv2(Config.segmentTestPath2,128,128)

    # Load models
    vgg19 = load_model(Config.vgg19Path)
    resnet = load_model(Config.resnet50Path)
    mobilenet = load_model(Config.mobilenetPath)
    modelList = [vgg19,resnet,mobilenet]

    # Load ensemble models
    enAvgW = Config.enAvgW
    enMaxW = Config.enMaxW
    enDLW = Config.enDLW
    enAvg = get_final_model(vgg19,resnet,mobilenet,enAvgW,'Avg')
    enMax = get_final_model(vgg19,resnet,mobilenet,enMaxW,'Max')
    enDL = get_final_model(vgg19,resnet,mobilenet,enDLW,'DL')

    resultPath = os.path.join(Config.resultsPath,'EnDL')

    # Evaluate all the model with dataset https://www.kaggle.com/grassknoted/asl-alphabet
    avgg19,_,cvgg19 = get_scores(ty1,tx1,vgg19,labelMap)
    aresnset,_,cresnet = get_scores(ty1,tx1,resnet,labelMap)
    amobilenet,_,cmobilenet = get_scores(ty1,tx1,mobilenet,labelMap)
    aenAvg,_,cenAvg = get_scores(ty1,tx1,enAvg,labelMap)
    aenMax,_,cenMax = get_scores(ty1,tx1,enMax,labelMap)
    aenDL,_,cenDL = get_scores(ty1,tx1,enDL,labelMap)

    # Evaluate all the model with dataset https://www.kaggle.com/datasets/danrasband/asl-alphabet-test
    avgg192,_,cvgg192 = get_scores(ty2,tx2,vgg19,labelMap)
    aresnset,_,cresnet2 = get_scores(ty2,tx2,resnet,labelMap)
    amobilenet2,_,cmobilenet2 = get_scores(ty2,tx2,mobilenet,labelMap)
    aenAvg2,_,cenAvg2 = get_scores(ty2,tx2,enAvg,labelMap)
    aenMax2,_,cenMax2 = get_scores(ty2,tx2,enMax,labelMap)
    aenDL2,_,cenDL2 = get_scores(ty2,tx2,enDL,labelMap)

    plot_confusion_matrix(cenDL2,labelMap,resultPath,show=False)

