# 3.22 - split training and test sets

# pandas is used for data manipulation and analysis
# numpy is used for numerical operations
import numpy as np
import pandas as pd
from colorama import Back, Style, init
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# initialize colorama
init()

# import csv to pandas dataframe
df: pd.DataFrame = pd.read_csv("./data/Data.csv")

# fetch raw features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

print(Back.GREEN + " raw X " + Style.RESET_ALL)
print(X)
print(Back.GREEN + " raw y " + Style.RESET_ALL)
print(y)

# use imputer to fill missing values in columns 1 and 2
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(Back.GREEN + " X after imputation " + Style.RESET_ALL)
print(X)


# use one-hot encoding to transform categorical features in column 0
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))

print(Back.GREEN + " X after one hot encoding " + Style.RESET_ALL)
print(X)


# use label encoding to transform y
le = LabelEncoder()
y = le.fit_transform(y)

print(Back.GREEN + " y after encoding " + Style.RESET_ALL)
print(y)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(Back.GREEN + " X_train " + Style.RESET_ALL)
print(X_train)

print(Back.GREEN + " X_test " + Style.RESET_ALL)
print(X_test)

print(Back.GREEN + " y_train " + Style.RESET_ALL)
print(y_train)

print(Back.GREEN + " y_test " + Style.RESET_ALL)
print(y_test)
