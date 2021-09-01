import pandas as pd
import csv
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

df = pd.read_csv("data.csv")

velocity = df["Velocity"].tolist()
escaped = df["Escaped"].tolist()

fig = px.scatter(x=velocity, y=escaped)
fig.show()

#adding the line of regression (best fit)
velocityarray = np.array(velocity)
escapedarray = np.array(escaped)

#slope and intercept
m, c = np.polyfit(velocityarray, escapedarray, 1)
y = []

for x in velocityarray:
    yvalue = (m*x) + c
    y.append(yvalue)

fig2 = px.scatter(x=velocityarray, y=escapedarray)
fig2.update_layout(shapes = [
    dict(
        type = "line",
        y0 = min(y),
        y1 = max(y),
        x0 = min(velocityarray),
        x1 = max(velocityarray),
    )
])
fig2.show()

#reshape the list
X = np.reshape(velocity, (len(velocity), 1))
Y = np.reshape(escaped, (len(escaped), 1))

lr = LogisticRegression()
lr.fit(X, Y)
plt.figure()
plt.scatter(X.ravel(), Y, color = "black", zorder = 20)

def model(x):
    return 1/(1+np.exp(-x))

#using the line formula
xtest = np.linspace(0, 100, 200)
chances = model((xtest*lr.coef_) + lr.intercept_).ravel()

#plotting the graph

plt.plot(xtest, chances, color = "red", linewidth = 3)
plt.axhline(y = 0, color = "k", linestyle = "-")
plt.axhline(y = 1, color = "k", linestyle = "-")
plt.axhline(y = 0.5, color = "b", linestyle = "-")
plt.axvline(x = xtest[23], color = "b", linestyle = "--")

plt.ylabel("Y")
plt.xlabel("X")
plt.xlim(0, 30)
plt.show()

velocity2 = float(input("Enter the velocity: "))
chancesescape = model((velocity2*lr.coef_) + lr.intercept_).ravel()[0]
if chancesescape <= 0.01:
    print("Velocity will not escape")
elif chancesescape >= 1:
    print("Velocity will escape")
elif chancesescape <= 0.5:
    print("Velocity might not escape")
else:
    print("Velocity might escape")
