# 02.02 - multiple linear regression

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from colorama import Back, Style, init
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

init()

df = pd.read_csv("./data/50_Startups.csv")

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

print(Back.WHITE + " raw - X " + Style.RESET_ALL)
print(X)
print(Back.WHITE + " raw - y " + Style.RESET_ALL)
print(y)

# check if any numeric feature has missing values
numeric_columns = [i for i in range(X.shape[1]) if i != df.columns.get_loc("State")]
X_numeric = X[:, numeric_columns].astype(float)
missing_values = np.isnan(X_numeric)
missing_value_features = np.where(np.any(missing_values, axis=0))[0]
print(Back.RED + " Features with missing values in numeric columns " + Style.RESET_ALL)
print(missing_value_features)


# encode categorical data
categorical_features = [df.columns.get_loc("State")]

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), categorical_features)],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X))

print(Back.BLUE + " categorical features encoded - X " + Style.RESET_ALL)
print(X)


# split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(Back.BLUE + " training set " + Style.RESET_ALL)
print(X_train)
print(y_train)

print(Back.BLUE + " test set " + Style.RESET_ALL)
print(X_test)
print(y_test)

# train the multiple linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Calculate R² scores
train_r2 = regressor.score(X_train, y_train)
test_r2 = regressor.score(X_test, y_test)

# Get numeric features (excluding one-hot encoded state columns)
numeric_features = [
    i for i in range(X_train.shape[1]) if i not in [0, 1, 2]
]  # Skip one-hot encoded columns
# Original numeric feature names
feature_names = [
    "RnD",
    "Administration",
    "Marketing",
]

# Create multiple 3D plots for different feature combinations
for i in range(len(numeric_features)):
    for j in range(i + 1, len(numeric_features)):
        fig = plt.figure(figsize=(10, 8))
        ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

        x1 = X_train[:, numeric_features[i]]
        x2 = X_train[:, numeric_features[j]]

        # Create mesh grid
        x1_surf, x2_surf = np.meshgrid(
            np.linspace(x1.min(), x1.max(), 100), np.linspace(x2.min(), x2.max(), 100)
        )

        # Create prediction matrix
        X_surf = np.zeros((x1_surf.size, X_train.shape[1]))
        X_surf[:, numeric_features[i]] = x1_surf.ravel()
        X_surf[:, numeric_features[j]] = x2_surf.ravel()

        # Predict surface
        y_surf = regressor.predict(X_surf).reshape(x1_surf.shape)

        # Plot
        ax.plot_surface(x1_surf, x2_surf, y_surf, alpha=0.3, color="blue")
        ax.scatter(x1, x2, y_train, color="red", marker="o")

        # Set labels using original feature names
        ax.set_xlabel(feature_names[i])
        ax.set_ylabel(feature_names[j])
        ax.set_zlabel("Profit")
        plt.title(
            f"Multiple Linear Regression: {feature_names[i]} vs {feature_names[j]}"
        )

        # Add R² values to the plot
        plt.figtext(
            0.95,
            0.05,
            f"R² (train): {train_r2:.4f}\nR² (test): {test_r2:.4f}",
            ha="right",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.8),
        )

        plt.savefig(
            f"./output/s02_02_multiple_linear_regression_{feature_names[i]}_{feature_names[j]}.png"
        )
        plt.close()

# predict the test set results
y_pred = regressor.predict(X_test)

# plot the y_test and y_pred
plt.figure(figsize=(10, 8))

# Main scatter plot
scatter = plt.scatter(
    y_test, y_pred, c=np.abs(y_test - y_pred), cmap="viridis", alpha=0.6
)
plt.colorbar(scatter, label="Absolute Error")

# Add perfect prediction line
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect Prediction")

# Add regression line
slope, intercept, r_value, p_value, std_err = stats.linregress(y_test, y_pred)
plt.plot(y_test, slope * y_test + intercept, "g-", label="Regression Line")

plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.title(f"Actual vs Predicted Profit\nR² = {r_value:.4f}")
plt.legend()

plt.tight_layout()
plt.savefig("./output/s02_02_actual_vs_predicted.png")
plt.close()
