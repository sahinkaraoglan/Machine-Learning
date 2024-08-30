from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np

#CREATE DATASET

x = np.sort(5*np.random.rand(80,1), axis = 0)
y = np.sin(x).ravel()
y[::5]+= 0.5*(0.5 - np.random.rand(16))

#plt.scatter(x,y)

reg_1 = DecisionTreeRegressor(max_depth=2)
reg_2 = DecisionTreeRegressor(max_depth=5)
reg_1.fit(x,y)
reg_2.fit(x,y)


x_test = np.arange(0,5,0.05)[:, np.newaxis]
y_pred_1 = reg_1.predict(x_test)
y_pred_2 = reg_2.predict(x_test)

plt.figure()
plt.plot(x, y, c = "red", label = "data")
plt.scatter(x, y, c = "red", label = "data")
plt.plot(x_test, y_pred_1, color = "blue", label = "Max Depth: 2", linewidth = 2)
plt.plot(x_test, y_pred_2, color = "green",label = "Max Depth: 5", linewidth = 2)
plt.xlabel("data")
plt.ylabel("target")
plt.legend()