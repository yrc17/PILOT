""" This is an example for the application of PILOT"""
from pilot import Pilot, Tree
import pandas as pd
import numpy as np

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

# initialize the model
model = Pilot.PILOT()

# fit the dataset
model.fit(X,y)

# make predictions
pred = np.mean((model.predict(x=X) - y) ** 2)
print(pred)


