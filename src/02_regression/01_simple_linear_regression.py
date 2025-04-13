# 02.01 - simple linear regression

import numpy as np
import pandas as pd
from colorama import Back, Style, init
from matplotlib import pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# initialize colorama
init()

df = pd.read_csv("./data/Salary_Data.csv")

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
print(X_train.shape)

print(Back.BLUE + " split into training and test sets - X_test " + Style.RESET_ALL)
print(X_test.shape)

print(Back.BLUE + " split into training and test sets - y_train " + Style.RESET_ALL)
print(y_train.shape)

print(Back.BLUE + " split into training and test sets - y_test " + Style.RESET_ALL)
print(y_test.shape)

# preprocessed data
print(Back.BLUE + " preprocessed data - X_train, y_train " + Style.RESET_ALL)
print(X_train)
print(y_train)

print(Back.BLUE + " preprocessed data - X_test, y_test " + Style.RESET_ALL)
print(X_test)
print(y_test)

# training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

print(Back.GREEN + " model trained! " + Style.RESET_ALL)
print(f"Coefficients: {regressor.coef_}")
print(f"Intercept: {regressor.intercept_}")

# predicting the test set results
y_pred = regressor.predict(X_test)

print(Back.GREEN + " predicted results " + Style.RESET_ALL)
print(
    pd.DataFrame(
        {
            "Years of Experience": X_test[:, 0],
            "Actual": y_test,
            "Predicted": y_pred,
            "Difference %": (y_test - y_pred) / y_test * 100,
        }
    )
)


# score: model's accuracy, meaning how much of the variance in the dependent variable
# is explained by the independent variable
print(f"Model score (R²): {regressor.score(X_test, y_test):.4f}")


# visualizing the training set results
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (Training set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.savefig("./output/s02_01_simple_linear_regression_training_set.png")

# visualizing the test set results
plt.clf()
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("Salary vs Experience (Test set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")

# Add R-squared score in bottom right
plt.text(
    0.95,
    0.05,
    f"R² = {regressor.score(X_test, y_test):.4f}",
    transform=plt.gca().transAxes,
    horizontalalignment="right",
    verticalalignment="bottom",
    bbox=dict(facecolor="white", alpha=0.8),
)

plt.savefig("./output/s02_01_simple_linear_regression_test_set.png")


print(
    f"Predicted salary for 22 years of experience: {regressor.predict(np.array([[22]]))[0]:.0f}"
)
