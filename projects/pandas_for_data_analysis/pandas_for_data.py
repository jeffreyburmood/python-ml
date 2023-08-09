""" This file contains a number of examples of the usee of pandas for managing data """

import pandas as pd

# how to read tabular data into pandas
orders = pd.read_table("http://bit.ly/chiporders")
print(orders.head())

user_cols = ["user_id", "age", "gender", "occupation", "zip_code"]
users = pd.read_table("http://bit.ly/movieusers", sep="|", header=None, names=user_cols)
print(users.head())

# how to select a Series from a Dataframe
