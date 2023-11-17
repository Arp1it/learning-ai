import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error


diabetes = datasets.load_diabetes()

# print(diabetes.keys())
# print(diabetes.data)
# print(diabetes.DESCR)

diabetes_X = diabetes.data
# diabetes_X = diabetes.data[:, np.newaxis, 2]
# print(diabetes_X)

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_Y_train = diabetes.target[:-30]
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)

diabetes_Y_predict = model.predict(diabetes_X_test)

print(f"Mean square error is {mean_squared_error(diabetes_Y_test, diabetes_Y_predict)}")
print(f"Weights {model.coef_}")
print(f"Intercept {model.intercept_}")

# plt.scatter(diabetes_X_test, diabetes_Y_test)
# plt.plot(diabetes_X_test, diabetes_Y_predict)
# plt.show()