print("REDES NEURONALES")
A=4
B=5
SUMA=4+5
RESTA=5-2
print(SUMA)
print(RESTA)

key = "senati"
password = input("introduce la contraseña: ")
if  key == password.lower():
      print("La contraseña coincide")
else:
      print("La contraseña no coincide")


import tensorflow as tf
print(tf.__version__)
import numpy as np

from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import joblib

import seaborn as sns

import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred,normalize=False,title=None):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Matriz de Confusión Normalizada'
        else:
            title = 'Matriz de Confusión sin Normalizar'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = unique_labels(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Matriz de Confusión Normalizada")
    else:
        print('Matriz de Confusión sin Normalizar')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.grid(linewidth=.0)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
            ha="center", va="center",
            color="white" if cm[i, j] > thresh else "black")
    #fig.tight_layout()
    plt.show()
    return ax

def saveFile(object_to_save, scaler_filename):
    joblib.dump(object_to_save, scaler_filename)

def loadFile(scaler_filename):
    return joblib.load(scaler_filename)

def plotHistogram(dataset_final):
    dataset_final.hist(figsize=(20,14), edgecolor="black", bins=40)
    plt.show()

def plotCorrelations(dataset_final):
    fig, ax = plt.subplots(figsize=(10,8))   # size in inches
    g = sns.heatmap(dataset_final.corr(), annot=True, cmap="YlGnBu", ax=ax)
    g.set_yticklabels(g.get_yticklabels(), rotation = 0)
    g.set_xticklabels(g.get_xticklabels(), rotation = 45)
    fig.tight_layout()
    plt.show()

# Funciones
def printBalanceo(data, field_grouping, data_name):
    print("\nCantidad de elementos por Clase en ", data_name, ":")
    target_count = data[field_grouping].value_counts()
    target_count.plot(kind='bar', title='Count (' + field_grouping + ')');

    print('Clase 0:', target_count[0], "({:.2%})".format(round(target_count[0] / (target_count[0]+target_count[1]), 2)))
    print('Clase 1:', target_count[1], "({:.2%})".format(round(target_count[1] / (target_count[0]+target_count[1]), 2)))
    print('Total  :', data.shape[0])


'''REDES NEURONALES'''
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


ruta_csv = "diabetes_data.csv"

data_diabetes=pd.read_csv(ruta_csv)
data_diabetes.head(10)

data_diabetes.info()

plotHistogram(data_diabetes)

printBalanceo(data_diabetes, 'PacienteDiabetico', 'DATASET ORIGINAL')



#Escalamiento/Normalización de variables
# Obteniendo valores y nombres de columnas por separado
dataset_values = data_diabetes.values 
dataset_columns = data_diabetes.columns

# Escalamiento/Normalización de Features (variables independintes X)
# StandardScaler: (x-u)/s
stdScaler = StandardScaler()
dataset_values[:,0:-1] = stdScaler.fit_transform(dataset_values[:,0:-1])

# Dataset final normalizado
dataset_final = pd.DataFrame(dataset_values,columns=dataset_columns, dtype=np.float64)

print ("\nDataset Final:")
dataset_final.head(10)


# Distribuciones de la data y Correlaciones
print("\n Histogramas:")
plotHistogram(dataset_final)

print("\n Correlaciones:")
plotCorrelations(dataset_final)


# Dividiendo el Dataset en sets de Training y Test
train, test = train_test_split(dataset_final, test_size =0.2, random_state = 1)


printBalanceo(train, 'PacienteDiabetico', 'Dataset final (Train)')


printBalanceo(test, 'PacienteDiabetico', 'Dataset final (Test)')

# Datos de Entrenamiento y Test (X vs Y)
# TRAIN
X_train = train.iloc[:, 0:-1].values #Numpy object
y_train = train.iloc[:, -1].values #Numpy object

# TEST
X_test = test.iloc[:, 0:-1].values #Numpy object
y_test = test.iloc[:, -1].values #Numpy object