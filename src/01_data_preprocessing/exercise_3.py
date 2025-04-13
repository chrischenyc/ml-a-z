# 01.exercise.3

import numpy as np
import pandas as pd
from colorama import Back, Style, init
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# initialize colorama
init()

df = pd.read_csv("./data/wine-quality-red.csv", sep=";")

print(Back.WHITE + " raw dataframe " + Style.RESET_ALL)
print(df)


X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(Back.BLUE + " raw - X " + Style.RESET_ALL)
print(X)

print(Back.BLUE + " raw - y " + Style.RESET_ALL)
print(y)

# fix missing values
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
imputer.fit(X)
X = imputer.transform(X)

print(Back.BLUE + " imputed - X " + Style.RESET_ALL)
print(X)

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
X_train[:, :] = sc.fit_transform(X_train[:, :])
X_test[:, :] = sc.transform(X_test[:, :])


# final preprocessed data
print(Back.GREEN + " final preprocessed data - X_train, y_train " + Style.RESET_ALL)
print(X_train)
print(y_train)

print(Back.GREEN + " final preprocessed data - X_test, y_test " + Style.RESET_ALL)
print(X_test)
print(y_test)
