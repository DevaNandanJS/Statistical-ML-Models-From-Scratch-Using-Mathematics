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

# Select features and target (Reduced to 2 features for 2D plotting)
x_train = data[['Age', 'Fare']].values
y_train = data['Survived'].values

m, n = x_train.shape
print(m,n)

#sigmoid
def sigmoid(z):
    return 1/ (1+ np.exp(-z))
 
#cost fuction 
def cost_function(x, y, w, b):
    m_local = x.shape[0]
    cost_sum = 0
    for i in range(m_local):
        z = np.dot(w, x[i]) + b
        g = sigmoid(z)
        g = np.clip(g, 1e-15, 1 - 1e-15)
        cost_sum += -y[i] * np.log(g) - (1-y[i]) * np.log(1-g)
    return (1/m_local) * cost_sum

def gradient_function(x, y, w, b):
    m_local, n_local = x.shape
    grad_w = np.zeros(n_local)
    grad_b = 0

    for i in range(m_local):
        z = np.dot(w, x[i]) + b
        g = sigmoid(z)
        err = g - y[i]
        for j in range(n_local):
            grad_w[j] += err * x[i, j]
        grad_b += err

    grad_w = (1/m_local) * grad_w
    grad_b = (1/m_local) * grad_b
    return grad_w, grad_b

def gradient_descent(x, y, alpha, iteration):
    w= np.zeros(n)
    b= 0

    for i in range(iteration):
        grad_w, grad_b = gradient_function(x, y, w, b)

        w = w - alpha * grad_w
        b = b - alpha * grad_b

        if i % 1000 == 0:
            print(f"Iteration {i}: cost is {cost_function(x, y, w, b)}")
    
    return w, b

def predict(x,w,b):
    preds= np.zeros(m)
    z= np.dot(x, w)+ b
    g= sigmoid(z)

    for i in range(m):
        if g[i]>= 0.5:
            preds[i]= 1
        else:
            preds[i] = 0
    return preds

#Running the  Model
iteration= 10000
alpha= 0.001

w, b= gradient_descent(x_train, y_train, alpha, iteration)

PREDICTIONS= predict(x_train,w,b)
accuracy= np.mean(PREDICTIONS== y_train)*100
print("Accuracy: ", accuracy)

# Plotting final
# Decision boundary: w[0]*x1 + w[1]*x2 + b = 0 => x2 = (-w[0]*x1 - b) / w[1]
slope = -w[0] / w[1]
intercept = -b / w[1]

xmin, xmax = x_train[:, 0].min() - 0.5, x_train[:, 0].max() + 0.5
ymin, ymax = x_train[:, 1].min() - 0.5, x_train[:, 1].max() + 0.5
xd = np.array([xmin, xmax])
yd = slope * xd + intercept

plt.figure(figsize=(10, 6))
plt.plot(xd, yd, 'k', ls='--', label='Decision Boundary')
plt.fill_between(xd, yd, ymin, color='tab:blue', alpha=0.1)
plt.fill_between(xd, yd, ymax, color='tab:orange', alpha=0.1)

plt.scatter(x_train[y_train == 0, 0], x_train[y_train == 0, 1], color='tab:blue', label='Did Not Survive', s=20, alpha=0.5)
plt.scatter(x_train[y_train == 1, 0], x_train[y_train == 1, 1], color='tab:orange', label='Survived', s=20, alpha=0.5)
plt.legend()
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Logistic Regression Decision Boundary (Titanic)')
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)
plt.show()