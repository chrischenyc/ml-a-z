# 3.11 - data preprocessing

# pandas is used for data manipulation and analysis
import pandas as pd

# import csv to pandas dataframe
dataset = pd.read_csv("./data/3/Data.csv")

print("--------------------------------dataset-----------------------------")
print(dataset)

print("--------------------------------info--------------------------------")
print(dataset.info())

print("--------------------------------describe----------------------------")
print(dataset.describe())

print("--------------------------------columns-----------------------------")
print(dataset.columns)

print("--------------------------------head--------------------------------")
print(dataset.head())

print("--------------------------------tail--------------------------------")
print(dataset.tail())
