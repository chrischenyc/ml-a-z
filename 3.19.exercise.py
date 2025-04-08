import numpy as np
import pandas as pd
from colorama import Back, Style, init
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# initialize colorama
init()

df: pd.DataFrame = pd.read_csv("./data/titanic.csv")

print(Back.GREEN + " raw columns " + Style.RESET_ALL)
print(df.columns.tolist())

# move 2nd column (label) to the end
df = df[
    [
        "PassengerId",
        "Pclass",
        "Name",
        "Sex",
        "Age",
        "SibSp",
        "Parch",
        "Ticket",
        "Fare",
        "Cabin",
        "Embarked",
        "Survived",
    ]
]

print(Back.GREEN + " reordered columns " + Style.RESET_ALL)
print(df.columns.tolist())

# raw features and labels
X_raw: pd.DataFrame = df.iloc[:, :-1]
y_raw: pd.Series = df.iloc[:, -1]

print(Back.GREEN + " X columns " + Style.RESET_ALL)
print(X_raw.columns.tolist())

print(Back.GREEN + " y column " + Style.RESET_ALL)
print(y_raw.name)

# transform categorical features with one-hot encoding
categorical_features = ["Pclass", "Sex", "Embarked"]
ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), categorical_features)],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X_raw))

# Get feature names automatically
feature_names = ct.get_feature_names_out()

# Label encoding
le = LabelEncoder()
y = le.fit_transform(y_raw)

# output the processed features and labels to a csv file
df_processed = pd.DataFrame(X, columns=feature_names)
df_processed["Survived"] = y

# save the processed data to a csv file
df_processed.to_csv("./data/titanic_processed.csv", index=False)
