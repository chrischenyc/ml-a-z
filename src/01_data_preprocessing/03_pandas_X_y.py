# 01.03 - pandas X and y

# pandas is used for data manipulation and analysis
import pandas as pd

# import csv to pandas dataframe
dataset = pd.read_csv("./data/Data.csv")


# select all rows and all columns except the last one as features
X = dataset.iloc[:, :-1].values

# select all rows and the last column as labels
y = dataset.iloc[:, -1].values

print(" X ")
print(X)

print(" y ")
print(y)
