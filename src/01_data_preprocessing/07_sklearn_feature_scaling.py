# 01.07 - feature scaling

import numpy as np
import pandas as pd
from colorama import Back, Style, init
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# initialize colorama
init()

# import csv to pandas dataframe
df: pd.DataFrame = pd.read_csv("./data/Data.csv")

# fetch raw features and labels
X = df.iloc[:, :-1].values
y = df.iloc[:, -1]

print(Back.WHITE + " raw - X " + Style.RESET_ALL)
print(X)
print(Back.WHITE + " raw - y " + Style.RESET_ALL)
print(y)

# use imputer to fill missing values in columns 1 and 2
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

print(Back.BLUE + " fill missing values with imputer - X " + Style.RESET_ALL)
print(X)


# use one-hot encoding to transform categorical features in column 0
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [0])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))

print(
    Back.BLUE
    + " transform categorical features with one-hot encoding - X "
    + Style.RESET_ALL
)
print(X)


# use label encoding to transform y
le = LabelEncoder()
y = le.fit_transform(y)

print(Back.BLUE + " transform with label encoding - y " + Style.RESET_ALL)
print(y)

# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

print(Back.BLUE + " split into training and test sets - X_train " + Style.RESET_ALL)
print(X_train)

print(Back.BLUE + " split into training and test sets - X_test " + Style.RESET_ALL)
print(X_test)

print(Back.BLUE + " split into training and test sets - y_train " + Style.RESET_ALL)
print(y_train)

print(Back.BLUE + " split into training and test sets - y_test " + Style.RESET_ALL)
print(y_test)

# feature scaling
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])

print(Back.BLUE + " scale features in training set - X_train " + Style.RESET_ALL)
print(X_train)

print(Back.BLUE + " scale features in test set - X_test " + Style.RESET_ALL)
print(X_test)

# final output

print(Back.GREEN + " preprocessed data - X_train, y_train " + Style.RESET_ALL)
print(X_train)
print(y_train)

print(Back.GREEN + " preprocessed data - X_test, y_test " + Style.RESET_ALL)
print(X_test)
print(y_test)
