import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter


class KNN:
    def __init__(self, k):
        self.k = k
    def fit(self,X,y):
        self.X_train = X
        self.y_train = y

    def predict(self,X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self,x):
        # compute the distance
        distances = [self._euclidian_distance(x,x_train) for x_train in self.X_train]
        # get the closest k
        k_indices= np.argsort(distances)[:self.k]
        k_closest_labels = [self.y_train[i] for i in k_indices]
        # majority vote
        most_common = Counter(k_closest_labels).most_common()
        return most_common[0][0]

    def _euclidian_distance(self, x1, x2):
        distance = np.sqrt(np.sum((x1 - x2) ** 2))
        return distance

if __name__ == '__main__':
    k = 5
    knn = KNN(k)
    iris = load_iris()
    X,y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state = 4)
    knn.fit(X_train,y_train)
    predictions = knn.predict(X_test)
    # print(predictions)
    # total_predictions = len(predictions)
    # correct_predictions = 0
    # for val1,val2 in zip(predictions,y_test):
    #     if val1 == val2:
    #         correct_predictions += 1
    # accuracy = correct_predictions/total_predictions
    accuracy = np.sum(predictions == y_test)/len(y_test)
    print(accuracy)


