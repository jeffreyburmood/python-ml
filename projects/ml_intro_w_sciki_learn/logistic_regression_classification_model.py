""" Logistic Regession Classification Model Example using the Iris dataset """

""" scikit-learn 4-srep modeling pattern

    Step 1: Import the class you plan to use

    Step 2: Instantiate the estimator (model)

    Step 3: Fit the model with data (aka model training)

    Step 4: Predict the response for a new observation
    
    """

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

iris = load_iris()
# store feature matrix in "X" - capitalized because X is a matrix
X = iris.data

# store response vector in "y" - lower case because y is a vector
y = iris.target

logreg = LogisticRegression()

logreg.fit(X, y)

X_new = [[3, 5, 4, 2], [5, 4, 3, 2]]
prediction = logreg.predict(X_new)

print(prediction)
