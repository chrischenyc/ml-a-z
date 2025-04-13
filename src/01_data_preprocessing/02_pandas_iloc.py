# 01.02 - pandas dataframe iloc

# pandas is used for data manipulation and analysis
import pandas as pd

# import csv to pandas dataframe
dataset = pd.read_csv("./data/Data.csv")


# examples of using iloc to select data
print(" iloc select first 3 rows and first 2 columns ")
print(dataset.iloc[0:3, 0:2])

print(" iloc select all rows and first 2 columns ")
print(dataset.iloc[:, 0:2])


print(" iloc select first 3 rows and all columns ")
print(dataset.iloc[0:3, :])

print(" iloc select all rows and all columns ")
print(dataset.iloc[:, :])


print(" iloc select last row and all columns  ")
print(dataset.iloc[-1:, :])

print(" iloc select first row and all columns  ")
print(dataset.iloc[:1, :])
