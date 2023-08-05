""" Evaluation a Classification Model 
    1. What are some commn evaluation procedures?
    2. What is the usage of classification accuracy?
    3. How does a confusion matrix describe the performance of a classifier?
    4. What metrics can be computed from a confusion matrix?
    5. How can you adjust classifier performance by changing the classification threashold?
    6. What is the purpose of the ROC curve?
    7. How does the Area under the curve (AUC) differ from classification accuracy? """

# Pima Indian Diabetes dataset
import pandas as pd

url = "/Users/jeffrey/SynologyDrive/projects/Python-Programming/Training/python-ml/data/diabetes.csv"
col_names = [
    "pregnant",
    "glucose",
    "bp",
    "skin",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "label",
]
pima = pd.read_csv(url)

print(pima.head())

# select the features to use and create X and y
feature_cols = ["Pregnancies", "Insulin", "BMI", "Age"]
X = pima[feature_cols]
y = pima.Outcome

# build the training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# train a logistic regression model
from sklearn.linear_model import LogisticRegression

logreg = LogisticRegression()
logreg.fit(X_train, y_train)

# make the class predictions for the testing set
y_pred_class = logreg.predict(X_test)

# calculate the accuracy
from sklearn import metrics

print(f"Model Accuracy: {metrics.accuracy_score(y_test, y_pred_class)}")

# null accuracy - predicting the most frequent class (in this case zero)
# examing the class distribution of the testing set
print(y_test.value_counts())

# calculate the percentage of ones
print(y_test.mean())

# calculate the percentage of zeros
print(1 - y_test.mean())

# calcluate null accuracy for binary classification problems coded as 0/1
print(max(y_test.mean(), 1 - y_test.mean()))

# calculate null accuracy for multi-class classification problems
print(y_test.value_counts().head(1) / len(y_test))

# comparing true and predicted response values
# conclusion - classification accuracy does not take into account the underlying distribution of response values
print(f"True: {y_test.values[0:25]}")
print(f"Pred: {y_pred_class[0:25]}")
