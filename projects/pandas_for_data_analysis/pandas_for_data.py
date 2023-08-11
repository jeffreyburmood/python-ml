""" This file contains a number of examples of the usee of pandas for managing data """

import pandas as pd

# how to read tabular data into pandas
orders = pd.read_table("http://bit.ly/chiporders")
print(orders.head())

user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
users = pd.read_table("http://bit.ly/movieusers", sep="|", header=None, names=user_cols)
print(users.head())

# how to select a Series from a Dataframe
ufo = pd.read_csv("http://bit.ly/uforeports")
print(type(ufo))
print(ufo.head())

city_bracket = ufo["City"]
print(type(city_bracket))
print(city_bracket.head())

city_dot = ufo.City
print(type(city_dot))
print(city_dot.head())

# create a new series in a dataframe
ufo["Location"] = ufo.City + ", " + ufo.State
print(ufo.head())
