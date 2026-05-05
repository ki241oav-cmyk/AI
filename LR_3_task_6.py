import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 3 + np.sin(X) + np.random.uniform(-0.5, 0.5, m).reshape(-1, 1)

def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)
    train_errors, val_errors = [], []
    for m_size in range(1, len(X_train)):
        model.fit(X_train[:m_size], y_train[:m_size])
        y_train_predict = model.predict(X_train[:m_size])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m_size], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="Навчальний набір")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Перевірочний набір")
    plt.legend()
    plt.title(title)
    plt.xlabel("Розмір навчального набору")
    plt.ylabel("RMSE")
    plt.axis([0, 80, 0, 3])
    plt.show()


plot_learning_curves(LinearRegression(), X, y, "Криві навчання (Лінійна модель)")

poly_pipeline = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(poly_pipeline, X, y, "Криві навчання (Поліном ступеня 10)")