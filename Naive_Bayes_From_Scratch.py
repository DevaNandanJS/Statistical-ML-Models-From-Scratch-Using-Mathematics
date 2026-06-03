import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score 

#loading the dataset
data= pd.read_csv("datasets/Breast Cancer.csv")
x= data.drop(columns= ["id", "diagnosis"])
y= data["diagnosis"]
 
print(data.head())

#splitting the data 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state = 42)

class NaiveBayes:
    def __init__(self):
        self.priors = {}
        self.class_cond_probs = {} 
    
    def fit (self, X,y):
        