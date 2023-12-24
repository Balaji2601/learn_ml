import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
class LinerRegression:
    def __init__(self,lr = 0.01, n_iterations = 100):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iterations):
            y_pred = np.dot(X,self.weights) + self.bias
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db
    def predict(self,X):
        y_pred = np.dot(X,self.weights) + self.bias
        return y_pred

def mse(y_test, predictions):
    return np.mean((y_test - predictions)**2)

if __name__ == '__main__':
    linear_reg = LinerRegression(lr = 0.1)
    X,y = datasets.make_regression(n_samples = 100, n_features = 1, noise = 20, random_state = 4)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=40)
    linear_reg.fit(X_train,y_train)
    predictions = linear_reg.predict(X_test)
    mean_square_error = mse(y_test,predictions)
    print(mean_square_error)

    y_prediction_line = linear_reg.predict(X)
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize = (8,6))
    m1 = plt.scatter(X_train, y_train, color = cmap(0.9), s = 10)
    m2 = plt.scatter(X_test, y_test, color = cmap(0.5), s = 10)
    plt.plot(X, y_prediction_line, color = 'black', linewidth = 2, label = 'Prediction')
    plt.show()