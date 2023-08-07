""" This is a basic Hello World for working with datasets """
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)

print(iris.target)

print(iris.target_names)

print(type(iris.data))

print(type(iris.target))

print(iris.data.shape)

print(iris.target.shape)

# store feature matrix in "X" - capitalized because X is a matrix
X = iris.data

# store response vector in "y" - lower case because y is a vector
y = iris.target
