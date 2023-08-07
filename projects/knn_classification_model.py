""" K Nearest Neighbor Classification Model Example using the Iris dataset """

""" scikit-learn 4-srep modeling pattern

    Step 1: Import the class you plan to use

    Step 2: Instantiate the estimator (model)

    Step 3: Fit the model with data (aka model training)

    Step 4: Predict the response for a new observation
    
    """

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

iris = load_iris()
# store feature matrix in "X" - capitalized because X is a matrix
X = iris.data

# store response vector in "y" - lower case because y is a vector
y = iris.target

# start with K=1
knn_1 = KNeighborsClassifier(n_neighbors=1)  # all other parameters are default

knn_1.fit(X, y)

prediction = knn_1.predict([[3, 5, 4, 2]])  # must be 2D array

print(prediction[0])

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
prediction = knn_1.predict(X_new)

print(prediction)

# now use K=5
knn_5 = KNeighborsClassifier(n_neighbors=5)  # all other parameters are default

knn_5.fit(X, y)

prediction = knn_5.predict(X_new)

print(prediction)
