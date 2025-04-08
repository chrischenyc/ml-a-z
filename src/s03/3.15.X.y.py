# 3.11 - data preprocessing

# pandas is used for data manipulation and analysis
import pandas as pd

# import csv to pandas dataframe
dataset = pd.read_csv("./data/Data.csv")


features = dataset.iloc[:, :-1].values
labels = dataset.iloc[:, -1].values

print(" features ")
print(features)

print(" labels ")
print(labels)
