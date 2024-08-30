import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

x = 4 * np.random.rand(100,1)
y = 2 + 3*x**2 + np.random.rand(100,1) #y = 2 + 3x^2

# plt.scatter(x, y)

poly_feat = PolynomialFeatures(degree=2)
x_poly = poly_feat.fit_transform(x)


poly_reg = LinearRegression()
poly_reg.fit(x_poly, y)


plt.scatter(x, y, color = "blue")

x_test = np.linspace(0, 4, 100).reshape(-1, 1)
x_test_poly = poly_feat.transform(x_test)
y_pred = poly_reg.predict(x_test_poly)

plt.plot(x_test, y_pred, color = "red")
plt.xlabel("X")
plt.ylabel("y")
plt.title("Polinom Regresyon Modeli")
