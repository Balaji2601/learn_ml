import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
def sigmoid(x):
    return 1/(np.exp(-x)+1)
class LogisticRegression:
    def __init__(self,lr = 0.1,n_iterations = 100):
        self.lr = lr
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
    def fit(self,X,y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(n_samples):
            linear_pred = np.dot(X, self.weights) + self.bias
            # y_pred = 1/(1+1/(np.exp(np.dot(X.T, self.weights) + self.bias)))
            y_pred = sigmoid(linear_pred)
            dw = (1/n_samples) * np.dot(X.T,(y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            self.weights = self.weights - self.lr * dw
            self.bias = self.bias - self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred

if __name__ == '__main__':
    logistic_reg = LogisticRegression()
    lbc = load_breast_cancer()
    X,y = lbc.data, lbc.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=40)
    logistic_reg.fit(X_train,y_train)
    y_pred = logistic_reg.predict(X_test)
    accuracy = (np.sum(y_pred == y_test))/len(y_test)
    print(accuracy)

