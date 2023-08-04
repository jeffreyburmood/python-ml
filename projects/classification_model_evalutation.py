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
