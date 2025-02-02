""" This file contains code related to encoding text fields in data """
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

# Loading and exploring data
df = pd.read_csv('../../data/titanic_train.csv', nrows=10)

y = df['Survived'] # target as a dataframe with one column

df_new = pd.read_csv('../../data/titanic_new.csv', nrows=10)

ohe = OneHotEncoder()

logreg = LogisticRegression(solver='liblinear', random_state=1)

print(df)

# using a text encoder
# OneHotEncoder - each full name is treated as a category (not recommended)
# CountVectorizer - each word in a name is treated independently (recommended)
#   - uses 1-dimensional input (Series)
#   - CountVectorizer converts text into a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
vect = CountVectorizer()
dtm = vect.fit_transform(df['Name'])
print(dtm)
print(vect.get_feature_names_out())

# build a document term matrix to better understand each of the terms
# CountVectorizer counted how many times each word appeared
print(pd.DataFrame(dtm.toarray(), columns=vect.get_feature_names_out()))

cols = ['Parch', 'Fare', 'Embarked', 'Sex', 'Name']
X = df[cols]
print(X)

# Name is a 1-dimensional input so not in brackets
ct = make_column_transformer((ohe, ['Embarked', 'Sex']), (vect, 'Name'), ('passthrough', ['Parch', 'Fare']))
print(ct.fit_transform(X))
print(ct.get_feature_names_out())

pipe = make_pipeline(ct, logreg)
print(pipe.fit(X, y))

X_new = df_new[cols]
print(pipe.predict(X_new))

# how to vectorize two text fields in the data
ct_text = make_column_transformer(
    (vect, 'Name'),
    (vect, 'Ticket'))