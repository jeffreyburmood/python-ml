""" This is a basic Hello World for working with datasets """
from sklearn.datasets import load_iris

iris = load_iris()

print(iris.feature_names)

print(iris.target)

print(iris.target_names)
