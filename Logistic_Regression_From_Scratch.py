import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# LOADING THE DATA
data = pd.read_csv('datasets/Titanic-Dataset.csv')

# DATA PREPARATION
# Fill missing Age values with the median
data['Age'] = data['Age'].fillna(data['Age'].median())
# Encode Sex: female = 1, male = 0
data['Sex'] = data['Sex'].map({'female': 1, 'male': 0})

# Select features and target
x_train = data[['Age', 'Fare', 'Pclass', 'Sex']].values
y_train = data['Survived'].values

m, n = x_train.shape
print(m,n)

#sigmoid
#def sigmoid(z):
#   return 1/ (1+ np.exp(-z))
 

#cost fuction 
#def cost_function(x, y, a, c, w, b):
