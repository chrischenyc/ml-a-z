# use simple linear regression to predict the price of COIN share price based on BTC price

import matplotlib.dates as mdates
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("./data/BTC_COIN_weekly.csv")
df["date"] = pd.to_datetime(df["date"])  # Convert date strings to datetime objects

print(df.head())

# create figure and axis objects
fig, ax1 = plt.subplots(figsize=(12, 6))

# plot BTC price on first y-axis
ax1.plot(df["date"], df["btc_price"], "b-", label="BTC Price")
ax1.set_xlabel("Date")
ax1.set_ylabel("BTC Price", color="b")
ax1.tick_params(axis="y", labelcolor="b")

# Format x-axis to show dates every 3 months
ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

# create second y-axis
ax2 = ax1.twinx()
ax2.plot(df["date"], df["coin_share_price"], "r-", label="COIN Price")
ax2.set_ylabel("COIN Price", color="r")
ax2.tick_params(axis="y", labelcolor="r")

# add title
plt.title("BTC and COIN Price")

# add legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

plt.tight_layout()  # Adjust layout to prevent label cutoff
plt.savefig("./output/s06_exec_01_BTC_COIN_price.png")


# training data preprocessing
X = df[["btc_price"]].values
y = df[["coin_share_price"]].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# training the simple linear regression model on the training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# plotting the training set results
plt.clf()
plt.scatter(X_train, y_train, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("BTC and COIN Price (Training set)")
plt.xlabel("BTC Price")
plt.ylabel("COIN Price")
plt.savefig("./output/s06_exec_01_BTC_COIN_price_training_set.png")

# plotting the test set results
plt.clf()
plt.scatter(X_test, y_test, color="red")
plt.plot(X_train, regressor.predict(X_train), color="blue")
plt.title("BTC and COIN Price (Test set)")
plt.xlabel("BTC Price")
plt.ylabel("COIN Price")
plt.savefig("./output/s06_exec_01_BTC_COIN_price_test_set.png")
