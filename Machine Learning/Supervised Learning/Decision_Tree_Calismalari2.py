from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

diabetes = load_diabetes()

x = diabetes.data #features
y = diabetes.target #target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#KARAR AĞACI REGRESYON MODELİ

tree_reg = DecisionTreeRegressor(random_state = 42)
tree_reg.fit(x_train, y_train)

y_pred = tree_reg.predict(x_test)

mse = mean_squared_error(y_test, y_pred)
print("mse: ",mse)

rmse = np.sqrt(mse)
print("rmse: ",rmse)

