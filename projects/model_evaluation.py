""" Model Evaluation Procedure to pick the best model for prediction 

    Procedure 1: Train and Test on the Entire Dataset
        - Train the model on the entire dataset
        - Test the model on the same dataset and evaluate how well we did by comparing the predicted 
            response values with the actual response values.
    
    Procedure 2: Train Test/Split
        - Split the data into training set and testing set
        - Train the model on the training set
        - Test the model on the testing set, and evaluate the results
    
    """

# Procedure 1

from sklearn.datasets import load_iris

iris = load_iris()
# store feature matrix in "X" - capitalized because X is a matrix
X = iris.data

# store response vector in "y" - lower case because y is a vector
y = iris.target

# Logistic Regression
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()

logreg.fit(X, y)

y_pred = logreg.predict(X)

# Classificartion accuracy
#   - Proportion of correct predictions
#   - Common evaluation metric for classification problems

from sklearn import metrics

logistic_accuracy = metrics.accuracy_score(y, y_pred)
print(logistic_accuracy)

# KNN (K=5)

from sklearn.neighbors import KNeighborsClassifier

knn5 = KNeighborsClassifier(n_neighbors=5)  # all other parameters are default

knn5.fit(X, y)

y_pred = knn5.predict(X)

knn5_accuracy = metrics.accuracy_score(y, y_pred)
print(knn5_accuracy)

# KNN (K=1)

knn1 = KNeighborsClassifier(n_neighbors=1)  # all other parameters are default

knn1.fit(X, y)

y_pred = knn1.predict(X)

knn1_accuracy = metrics.accuracy_score(y, y_pred)
print(knn1_accuracy)

# Procedure 2

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=4)

# logistic regression
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
logistic_accuracy = metrics.accuracy_score(y_test, y_pred)
print(logistic_accuracy)

# KNN = 5
knn5.fit(X_train, y_train)

y_pred = knn5.predict(X_test)
knn5_accuracy = metrics.accuracy_score(y_test, y_pred)
print(knn5_accuracy)

# KNN = 1
knn1.fit(X_train, y_train)

y_pred = knn1.predict(X_test)
knn1_accuracy = metrics.accuracy_score(y_test, y_pred)
print(knn1_accuracy)

# different values of K
k_range = range(1, 26)
scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.accuracy_score(y_test, y_pred))

# plot the scores
import matplotlib.pyplot as plt

plt.plot(k_range, scores)
plt.xlabel("Value of k")
plt.ylabel("Testing Accuracy")
plt.show()

# Making predictions on out-of-sample data
# choose the best value for k
knn = KNeighborsClassifier(n_neighbors=11)
# use the full dataset
knn.fit(X, y)
y_pred = knn.predict([[3, 5, 4, 2]])
print(y_pred)
