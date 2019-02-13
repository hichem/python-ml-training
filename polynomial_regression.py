# Polynomial Regression

# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('data/Position_Salaries.csv')

print("Dataset: ", dataset)

# Get X and y
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

print("X=", X)
print("y=", y)

# Visualising the Dataset
plt.scatter(X,y,color='red')
plt.title=('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,y)


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
X_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

# Visualising the Linear Regression results
plt.scatter(X,y,color='green')
plt.plot(X,lin_reg.predict(X),color='green')
plt.title=('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# Visualising the Polynomial Regression results
plt.scatter(X,y,color='green')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.title=('Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


# Visualising the Polynomial Regression results (for higher resolution and smoother curve)


# Predicting a new result with Linear Regression
value = 2.5
vector = np.linspace(1.5, 8.5, 22)
vector = vector.reshape(-1,1)
print(vector)
y_pred = lin_reg2.predict(vector)
#print("Linear prediction of %s: %s" % (value, y_lin_pred));
plt.scatter(X,y,color='red')
plt.plot(X,lin_reg2.predict(X_poly),color='blue')
plt.plot(vector,y_pred,color='yellow')
plt.title=('Estimated Salary vs Position')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

# Predicting a new result with Polynomial Regression

