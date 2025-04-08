# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# Load the dataset
dataset = pd.read_csv("./data/pima-indians-diabetes.csv.csv")


print("--------------------------------info----------------------------")
print(dataset.info())

# Identify missing data (assumes that missing data is represented as NaN)
missing_values = dataset.isnull().sum()

# Print the number of missing entries in each column
print("--------------------------------missing values----------------------------")
print(missing_values)

# Configure an instance of the SimpleImputer class
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# Fit the imputer on the DataFrame
imputer.fit(dataset.iloc[:, :-1])

# Apply the transform to the DataFrame
features = imputer.transform(dataset.iloc[:, :-1])
labels = dataset.iloc[:, -1]

# Print your updated matrix of features
print(
    "--------------------------------updated matrix of features----------------------------"
)
print(features)

print(
    "--------------------------------updated matrix of labels----------------------------"
)
print(labels)
