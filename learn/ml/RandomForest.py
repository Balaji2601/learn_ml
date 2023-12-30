import numpy as np
from sklearn import datasets
from DecisionTree import DecisionTree
from collections import Counter
from sklearn.model_selection import train_test_split
class RandomForest:
    def __init__(self, n_trees = 10, max_depth = 10, min_samples_split = 2, n_feature = None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_feature
        self.trees = []

    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(min_samples_split=self.min_samples_split, max_depth=self.max_depth, n_features=self.n_features)
            X_sample, y_sample = self._boot_strap(X,y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
    def _boot_strap(self,X,y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common_label = counter.most_common(1)[0][0]
        return most_common_label
    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # print(predictions.shape)
        # print(predictions)
        tree_pred = np.swapaxes(predictions,0,1)
        # tree_pred = predictions.T
        preds = np.array([self._most_common_label(pred) for pred in tree_pred])
        return preds

if __name__ == '__main__':
    data = datasets.load_breast_cancer()
    X,y = data.data, data.target
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1234)
    def accuracy(y_true, y_pred):
        accu = np.sum(y_true == y_pred) / len(y_true)
        return accu

    clf = RandomForest(n_trees=20)
    clf.fit(X_train,y_train)
    predictions = clf.predict(X_test)
    acc = accuracy(y_test,predictions)
    # print(acc)
    print(acc)