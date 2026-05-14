import pandas as pd
import numpy as np 
from collections import Counter
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv('datasets/Breast Cancer.csv')
print(df.head())

#splitting the data
# Drop id, diagnosis, and any all-NaN columns (e.g. 'Unnamed: 32')
x= df.drop(columns=['id','diagnosis']).dropna(axis=1, how='all').values
y= df['diagnosis'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2, random_state=2)

# Feature scaling — z-score standardisation (from scratch)
# Computed only on training data; same mean/std applied to test data
mean = np.mean(x_train, axis=0)
std  = np.std(x_train, axis=0)
std[std == 0] = 1          # avoid division by zero for constant features
x_train = (x_train - mean) / std
x_test  = (x_test  - mean) / std

#plotting data
plt.scatter(x_train[y_train== 'B', 0], x_train[y_train== 'B', 1], color='blue', label='Benign')
plt.scatter(x_train[y_train== 'M', 0], x_train[y_train== 'M', 1], color= 'red', label='Malignant')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.show()

#distance metric
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a-b)**2))

#implementing KNN from scratch
class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, new_points):
        predictions = [self.predict_class(new_point) for new_point in new_points]
        return np.array(predictions)
    
    def predict_class(self, new_point):
        distances = [euclidean_distance(point, new_point) for point in self.X_train]

        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_nearest_indices]

        most_common = Counter(k_nearest_labels).most_common(1)[0][0]

        return most_common

#execution
knn = KNN(7)
knn.fit(x_train, y_train)
predictions = knn.predict(x_test)
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

#plotting results
plt.scatter(x_test[predictions == 'B', 0], x_test[predictions == 'B', 1], color='tab:green', label='Predicted Benign', marker='x')
plt.scatter(x_test[predictions == 'M', 0], x_test[predictions == 'M', 1], color='tab:red', label='Predicted Malignant', marker='x')
plt.xlabel('Radius Mean')
plt.ylabel('Texture Mean')
plt.legend()
plt.show()