""" performing basic data quality checks """

from sklearn.datasets import fetch_california_housing
import pandas as pd
import re

data = fetch_california_housing()

# convert the data to a pandas dataframe
df = pd.DataFrame(data.data, columns=data.feature_names)

# add target column
df['MedHouseVal'] = data.target

# print(data.DESCR)

# print(df.info)

print(df.describe())

############################
# Check for missing values
############################
missing_values = df.isnull().sum()
print("Missing Values:")
print(missing_values)

############################
# Identify duplicate records
############################
duplicate_rows = df[df.duplicated()]
print("Duplicate rows:")
print(duplicate_rows)

##################
# Check datatypes
##################
data_types = df.dtypes
print("Data types:")
print(data_types)

#####################
# Check for outliers
#####################
columns_to_check = ["MedInc", "AveRooms", "AveBedrms", "Population"]

# find records with outliers
def find_outliers(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
    return outliers

# fnd records with outliers for each column
outliers_dict = dict()

for column in columns_to_check:
    outliers_dict[column] = find_outliers(df, column)

# print the records with outliers for each column
for column, outliers in outliers_dict.items():
    print(f"Outliers in {column}:")
    print(outliers)
    print("\n")

##########################
# Validate numeric ranges
##########################
# Check numeric range for MedIc column
valid_range = (0, 16)
value_range_check = df[~df['MedInc'].between(*valid_range)]
print("Valid range check:")
print(value_range_check)

##########################
# Cross-column dependency
##########################
# AveRooms should not be smaller than AveBedrooms
invalid_data = df[df['AveRooms'] < df['AveBedrms']]
print("invalid records (AveRooms < AveBedrooms):")
print(invalid_data)

##################################
# Check for consistent data entry
##################################
# make sure all Date entries are in YYYY-MM-DD format
data = {'Date': ['2023-10-19', '2023-11-15', '23-10-2023', '2023/10/19', '2023-10-30']}
df = pd.DataFrame(data)

# define the expected format
date_format_pattern = r'^\d{4}-\d{2}-\d{2}$'

# function to check of the expected format pattern matches the entry
def check_date_format(date_str, format_pattern):
    return re.match(format_pattern, date_str) is not None

# apply the format check to the Date column
date_format_check = df['Date'].apply(lambda x: check_date_format(x, date_format_pattern))

# identify any entries that do not adhere to the format pattern
non_adherent_dates = df[~date_format_check]

if not non_adherent_dates.empty:
    print("Entries that do not follow the date format pattern:")
    print(non_adherent_dates)
else:
    print("All dates adhere to the date format pattern")




