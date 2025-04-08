# 3.17 - missing data

# pandas is used for data manipulation and analysis
# numpy is used for numerical operations
import numpy as np
import pandas as pd

# sklearn is used for machine learning
from sklearn.impute import SimpleImputer

# import csv to pandas dataframe
dataset = pd.read_csv("./data/3/Data.csv")


features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

print(
    "--------------------------------features before imputation----------------------------"
)
print(features)


# imputer is used to impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# fit imputer to features with numeric values
imputer.fit(features[:, 1:3])

# transform features with numeric values
features[:, 1:3] = imputer.transform(features[:, 1:3])

print(
    "--------------------------------features after imputation----------------------------"
)
print(features)
