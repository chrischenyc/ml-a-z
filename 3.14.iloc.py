# 3.11 - data preprocessing

# pandas is used for data manipulation and analysis
import pandas as pd
# numpy is used for numerical operations
import numpy as np
# matplotlib is used for data visualization
import matplotlib.pyplot as plt

# import csv to pandas dataframe
dataset = pd.read_csv("./data/3/Data.csv")


# examples of using iloc to select data
print("--------------------------------iloc select first 3 rows and first 2 columns--------------------------------")
print(dataset.iloc[0:3, 0:2])

print("--------------------------------iloc select all rows and first 2 columns--------------------------------")
print(dataset.iloc[:, 0:2])


print("--------------------------------iloc select first 3 rows and all columns--------------------------------")
print(dataset.iloc[0:3, :])

print("--------------------------------iloc select all rows and all columns--------------------------------")
print(dataset.iloc[:, :])


print("--------------------------------iloc select last row and all columns --------------------------------")
print(dataset.iloc[-1: , :])

print("--------------------------------iloc select first row and all columns --------------------------------")
print(dataset.iloc[:1, :])
