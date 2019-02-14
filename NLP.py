# Natural Language Processing

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Importing the dataset
dataset = pd.read_csv('data/Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# Cleaning the texts
import re     # regular expression library
import nltk   # native language toolkit

#Download nltk stop words dictionary
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


#Extract meaningful / relevant words by removing stop words of english language
ps = PorterStemmer()
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=0)


# Classifiers and predictions
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

def Classifier(X_train, X_test, y_train, y_test):
    DT = DecisionTreeClassifier()
    DT.fit(X_train, y_train)
    yDT = DT.predict(X_test)
    cmDT = confusion_matrix(y_test, yDT)
    print("Decision Tree Confusion Matrix: ", cmDT)
    print('Accuracy Score of Decision Tree: {0:.3f}'.format(accuracy_score(y_test, yDT)))
    print('Precision Score of Decision Tree: {0:.3f}'.format(accuracy_score(y_test, yDT)))
    
    KNN = KNeighborsClassifier()
    KNN.fit(X_train, y_train)
    yKNN = KNN.predict(X_test)
    cmKNN = confusion_matrix(y_test, yKNN)
    print("Decision Tree Confusion Matrix: ", cmKNN)
    print('Accuracy Score of Decision Tree: {0:.3f}'.format(accuracy_score(y_test, yKNN)))
    print('Precision Score of Decision Tree: {0:.3f}'.format(accuracy_score(y_test, yKNN)))
    
    NB = GaussianNB()
    NB.fit(X_train, y_train)
    yNB = NB.predict(X_test)
    cmNB = confusion_matrix(y_test, yNB)
    print("Decision Tree Confusion Matrix: ", cmNB)
    print('Accuracy Score of Decision Tree: {0:.3f}'.format(accuracy_score(y_test, yNB)))
    print('Precision Score of Decision Tree: {0:.3f}'.format(accuracy_score(y_test, yNB)))

Classifier(X_train, X_test, y_train, y_test)

# Dimensionality reduction PCA
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
sum(pca.explained_variance_ratio_)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

Classifier(X_train_pca, X_test_pca, y_train, y_test)
