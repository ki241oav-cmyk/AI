import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
input_file = 'data_regr_4.txt' 
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]
num_training = int(0.8 * len(X))
X_train, y_train = X[:num_training], y[:num_training]
X_test, y_test = X[num_training:], y[num_training:]
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)
y_test_pred = regressor.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='green', label='Actual data')
plt.plot(X_test, y_test_pred, color='black', linewidth=3, label='Regression line')
plt.title(f'Regression model for {input_file}')
plt.xlabel('Input Variable')
plt.ylabel('Output Variable')
plt.legend()
plt.show()
print("Linear regressor performance:")
print("Mean absolute error =", round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error =", round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("R2 score =", round(sm.r2_score(y_test, y_test_pred), 2))