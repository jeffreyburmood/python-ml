""" This file contains examples related to K-fold cross-validation to
    estimate the likely performance of a model.
     
    Better than the train/test split accuracy method. """

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# read in the iris data directly
iris = load_iris()

X = iris.data
y = iris.target

# split up the train/test data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=4)

# check the classification accuracy
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(f"train/test split accuracy = {metrics.accuracy_score(y_test, y_pred)}")
