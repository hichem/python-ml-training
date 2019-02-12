# Artificial Neural Network

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
print("Dataset: ", dataset)

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
labelencoder_X_1 = LabelEncoder()
X[:,1] = labelencoder_X_1.fit_transform(X[:,1])
labelencoder_X_2 = LabelEncoder()
X[:,2] = labelencoder_X_2.fit_transform(X[:,2])

from sklearn.preprocessing import OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:,1:];


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
# fit determines the parameters of the transformation to apply to data to normalize it
# tranform just applys the transform on any subsequent set of data to normalize it
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Part 2 - Now let's make the ANN

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer. We have as input 11 features, and we will be using 6 neurons on the hidden layer
# In general, we use the relu function in hidden layers
classifier.add(Dense(output_dim=6, init='uniform', activation='relu', input_dim=11))

# Adding the second hidden layer. The input to this second hidden layer is the 6 outputs from the first 6-neurons layers.
# As number of input and output is the same, we don't have to specify argument input_dim=6
classifier.add(Dense(output_dim=6, init='uniform', activation='relu'))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy')

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size=100, nb_epoch=20)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)


# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

