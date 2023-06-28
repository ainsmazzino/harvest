import numpy as np
import pandas as pd
import quandl
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = quandl.get("WIKI/FB")
print(df.head())

df = df[['Adj. Close']]
print(df.head())

forecast_out = 7

df['Prediction'] = df[['Adj. Close']].shift(-forecast_out)
print(df.tail())

X = np.array(df.drop(['Prediction'],1))
X = X[:-forecast_out]
print(X)

y = np.array(df['Prediction'])
y = y[:-forecast_out]
print(y)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = LinearRegression()
lr.fit(x_train, y_train)

lr_con = lr.score(x_test, y_test)
print("Accuracy : ", lr_con)

x_forecast = np.array(df.drop(['Prediction'],1))[-forecast_out:]
print(x_forecast)

lr_prediction = lr.predict(x_forecast)
print(lr_prediction)

DF = pd.DataFrame(lr_prediction)
DF.to_csv("hola.csv")

dff = pd.read_csv('hola.csv')
input()