import pandas as pd
import numpy
import matplotlib.pyplot as plt

#loading the csv
data= pd.read_csv('datasets/student_scores.csv')
print(data.head())
print(data.shape)

#plotting the data
plt.scatter(data.Hours, data.Scores, color='blue')
plt.title('Hours vs Scores')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()

#Defining loss function, We dont use this and its just for learning and understanding purpose
def loss_function(m, b, points):
    total_error= 0
    for i in range(len(points)):
        x= points.iloc[i].Hours
        y= points.iloc[i].Scores
        total_error= total_error + (y - (m*x)+b )**2
    total_error / len(points)
    return total_error

#Defining gradient descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient= 0
    b_gradient= 0

    n= len(points)

    for i in range(n):
        x= points.iloc[i].Hours
        y= points.iloc[i].Scores

        m_gradient= m_gradient - (2/n) * (y- (m_now*x + b_now)) * x
        b_gradient= b_gradient - (2/n) * (y- (m_now*x + b_now))
    
    m = m_now - L * m_gradient
    b = b_now - L * b_gradient
    return m, b

#Exicution
m= 0
b= 0
l= 0.0001
epochs= 1000

for i in range(epochs):
    if i % 50 == 0:
        print(f"epoch {i} loss: {loss_function(m,b,data)}")

    m,b= gradient_descent(m,b, data, l)

print(f"the weight is {m} and the bias is {b}")

#plotting the regression line
plt.scatter(data.Hours, data.Scores, color='black') #grph points

plt.plot(data.Hours, m*data.Hours + b, color='red') #tend line

plt.title('Hours vs Scores with Regression Line')
plt.xlabel('Hours Studied')
plt.ylabel('Scores Obtained')
plt.show()

