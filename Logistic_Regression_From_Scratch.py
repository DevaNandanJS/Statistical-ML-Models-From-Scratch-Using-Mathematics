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
def sigmoid(z):
    return 1/ (1+ np.exp(-z))
 
#cost fuction 
def cost_function(x, y, a, c, w, b):
    cost_sum= 0
    for i in range(m):
        z= np.dot(w, x[i]) + b
        g= sigmoid(z)

        cost_sum+= -y[i] * np.log(g) - (1-y[i]) * np.log(1-g[i])
        
    return 1/m * cost_sum

def gradient_function(x,y,w,b):
    grad_w= np.zeros (n)
    grad_b= 0

    for i in range(m):
        z= n.dot(w, x[i]) + b
        g= sigmoid(z)

        for j in range(n):
            grad_w+= (g-y[i]) * x[i,j]
        grad_b+= g-y[i]

    grad_w= grad_w * 1/m
    grad_b= grad_b * 1/m

    return grad_w, grad_b

def gradient_descent(x, y, w, b, alpha, iteration):
    w= np.zeros(n)
    b= 0

    for i in range(iteration):
        
        
