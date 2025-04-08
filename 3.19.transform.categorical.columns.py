# 3.19 - transform categorical columns

# pandas is used for data manipulation and analysis
# numpy is used for numerical operations
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# import csv to pandas dataframe
dataset = pd.read_csv("./data/3/Data.csv")


features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

print("----------------------features before preprocessing------------------")
print(features)
print("----------------------labels before preprocessing------------------")
print(labels)

# imputer is used to impute missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")

# fit imputer to features with numeric values
imputer.fit(features[:, 1:3])

# transform features with numeric values
features[:, 1:3] = imputer.transform(features[:, 1:3])

print("----------------------features after imputation------------------")
print(features)


# encoding categorical data
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])], remainder="passthrough"
)
features = np.array(ct.fit_transform(features))

print("-----------------features after one hot encoding------------------")
print(features)


# encoding labels
le = LabelEncoder()
labels = le.fit_transform(labels)

print("-----------------labels after encoding------------------")
print(labels)
