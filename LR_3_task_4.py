import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target

Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.5, random_state=0)

regr = linear_model.LinearRegression()
regr.fit(Xtrain, ytrain)

ypred = regr.predict(Xtest)

print("Коефіцієнти регресії (weights):", regr.coef_)
print("Вільний член (intercept):", round(regr.intercept_, 2))
print("Коефіцієнт детермінації R2:", round(r2_score(ytest, ypred), 4))
print("Середня абсолютна помилка (MAE):", round(mean_absolute_error(ytest, ypred), 2))
print("Середньоквадратична помилка (MSE):", round(mean_squared_error(ytest, ypred), 2))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(ytest, ypred, edgecolors=(0, 0, 0), alpha=0.7)
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4) 
ax.set_xlabel('Виміряно (Actual)')
ax.set_ylabel('Передбачено (Predicted)')
ax.set_title('Діагностика лінійної регресії (Diabetes Dataset)')
plt.show()