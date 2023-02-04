""" This is an example for the application of PILOT"""
from pilot import Pilot
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# import the dataset
path = "https://archive.ics.uci.edu/ml/machine-learning-databases/00514/Bias_correction_ucl.csv"
temp = pd.read_csv(path)
# feature transformation
temp = temp.dropna()
temp["Date"] = pd.to_datetime(temp["Date"])
temp["Day"] = temp["Date"].dt.weekday
temp["Month"] = temp["Date"].dt.month
temp["Year"] = temp["Date"].dt.year
temp.drop("Date", axis=1, inplace=True)
yr = {2013: 1, 2014: 2, 2015: 3, 2016: 4, 2017: 5}
temp["Year"] = temp["Year"].map(yr)
temp.station = temp.station.astype(int)

# define the predictors and the response
X, y = (
    temp.drop(["Next_Tmax", "Next_Tmin"], axis=1).values,
    temp["Next_Tmax"].values,
)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# initialize the model
model = Pilot.PILOT()

# fit on the training data
model.fit(X_train, y_train)

# make predictions and compute mse
pred_train = model.predict(x=X_train) 
mse_train = np.mean((pred_train - y_train) ** 2)
print(f"The mse on the training data is {mse_train}.")
pred_test = model.predict(x=X_test) 
mse_test = np.mean((pred_test - y_test) ** 2)
print(f"The mse on the test data is {mse_test}.")

# make prediction v.s. response plots on the training data
plt.scatter(pred_train, y_train, alpha = 0.3, s = 15)
# make the reference line such that pred = true response
x_grid = np.linspace(start = 18, stop = 38, num = 401)
plt.plot(x_grid, x_grid, linestyle = 'dashed', color = 'r')
plt.title('Prediction v.s. Response on Training Data')
plt.xlabel('Prediction')
plt.ylabel('y_train')
plt.savefig('train_scatter.jpg')
plt.close()

# make prediction v.s. response plots on the test data
plt.scatter(pred_test, y_test, alpha = 0.3, s = 15)
# make the reference line such that pred = true response
plt.plot(x_grid, x_grid, linestyle = 'dashed', color = 'r')
plt.title('Prediction v.s. Response on Test Data')
plt.xlabel('Prediction')
plt.ylabel('y_test')
plt.savefig('test_scatter.jpg')
plt.close()