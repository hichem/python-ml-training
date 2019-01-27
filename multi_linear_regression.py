# Multiple linear regression

# Importing the libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset

dataset = pd.read_csv('50_Startups.csv')


# dimensions

dataset.shape

# description statistique

dataset.describe()

# scatterpolt des variables pris deux à deux
# uniquement les variables quantitatives 

pd.tools.plotting.scatter_matrix(dataset.select_dtypes(exclude=['object']))

# les variables X et y

X = dataset.iloc[:,0:4].values
y = dataset.iloc[:,4].values


# Encoding categorical data
# Encoding the Independent Variable

from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,3] = labelencoder_X.fit_transform(X[:,3])
from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()

# éviter le piège de la variable Dummy

X = X[:,1:]

# Splitting the dataset into the Training set and Test set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,\
                                                    random_state=0)

# Fitting Multiple linear regression to the training set

# importer la classe LinearRegression du module linear_model de la librairie Scikit-learn



# Création de l'objet


# Utiliser la méthode fit de la classe LinearRegerssion



# predicting the test set results



# Building the optimal model using Backward elimination
# Importing statsmodels library


# ajouter une colonne formée par des 1 à X (première colonne)



# créer une nouvelle matrice qui va contenir les features optimaux


# création d'un objet regressor de la classe OLS (Ordinary Least Squares)
# C'est la méthode des moindres carrées ordinaires



# Summary regressor_OLS




