import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

def read_csv(models, file):
    """
    Function to read from csv files, and update the dictionary
    :param models: dictionary of details of models
    :param file: csv file
    :return: an updated dictionary
    """
    with open(file, mode='r') as file:
        rows = list(csv.reader(file))
        for row in rows:
            if row != []:
                #Update dictionary with {models: model's name, val_loss: loss, val_acc: accuracy, model: Path to the model}
                models = update_models(models, row[0], float(row[1]), float(row[2]), row[3])

    return models

def update_models(models, modelName, val_loss, val_acc, modelFile=None):
    models[modelName] = {"val_loss": val_loss, "val_acc": val_acc, "model": modelFile}
    return models


def update_results(models, path, graph=True):
    """
    Create a csv file, record the dictionary values into the csv file, plot graphs for val_loss and val_acc,
    save the graphs and csv file into given folder
    :param models: a dictionary with model's loss, accuracy and path
    :param path: directory to save csv and graphs to
    :param graph: Boolean to plot graphs
    :return:
    """
    csvFilename = os.path.join(path, 'Models.csv')
    Path(path).mkdir(parents=True, exist_ok=True)

    nameList, accList, lossList, modelList = [], [], [], []

    for name in models:
        nameList.append(name)
        accList.append(models[name]['val_acc'])
        lossList.append(models[name]['val_loss'])
        modelList.append(models[name]['model'])

    a_file = open(csvFilename, "w")

    writer = csv.writer(a_file)

    for key, value in models.items():
        writer.writerow([key, value['val_loss'], value['val_acc'], value['model']])
    a_file.close()

    if graph:
        plt.figure(figsize=(20, 5))
        accG = sns.barplot(x=nameList, y=accList, palette='mako')
        accG.set(xlabel='Model Name', ylabel='Accuracy', title='Accuracy Graph', ylim=[0.9, 1])
        plt.savefig(os.path.join(path, 'acc_graph.jpg'))

        plt.figure(figsize=(20, 5))
        lossG = sns.barplot(x=nameList, y=lossList, palette='rocket')
        lossG.set(xlabel='Model Name', ylabel='Loss', title='Loss Graph')
        plt.savefig(os.path.join(path, 'lose_graph.jpg'))

def plot_results(models):
    # Plot accuracy and loss graphs
    nameList, accList, lossList = [], [], []
    for name in models:
        nameList.append(name)
        accList.append(models[name]['val_acc'])
        lossList.append(models[name]['val_loss'])

    plt.figure(figsize=(20, 5))
    accG = sns.barplot(x=nameList, y=accList, palette='mako')
    accG.set(xlabel='Model Name', ylabel='Accuracy', title='Accuracy Graph', ylim=(0.98, 1))

    plt.figure(figsize=(20, 5))
    lossG = sns.barplot(x=nameList, y=lossList, palette='rocket')
    lossG.set(xlabel='Model Name', ylabel='Loss', title='Loss Graph', ylim=(0.0, 0.1))