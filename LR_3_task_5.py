import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, m).reshape(-1, 1)
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

print(f"Лінійна модель: y = {lin_reg.coef_[0][0]:.4f}x + {lin_reg.intercept_[0]:.4f}")
print(f"Поліноміальна модель (deg=2): y = {poly_reg.coef_[0][1]:.4f}x^2 + {poly_reg.coef_[0][0]:.4f}x + {poly_reg.intercept_[0]:.4f}")

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=15, label='Дані (Варіант 9)')
plt.plot(X, y_pred_lin, color='red', label='Лінійна регресія')
plt.plot(X, y_pred_poly, color='green', linewidth=2, label='Поліноміальна регресія (ступінь 2)')
plt.title("Самостійна побудова регресії")
plt.xlabel("x1")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()