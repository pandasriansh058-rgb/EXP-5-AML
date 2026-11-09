import numpy as np
import matplotlib.pyplot as plt


# ---------------- LASSO REGRESSION ----------------

def lasso_regression_gd(X1, X2, y, learning_rate, n_iter, lam):

    n = len(y)

    w1 = 0.0
    w2 = 0.0
    b = 0.0

    cost_history = []

    for i in range(n_iter):

        # Prediction
        y_pred = w1 * X1 + w2 * X2 + b

        # Lasso Cost Function
        cost = (
            (1 / n) * np.sum((y_pred - y) ** 2)
            + lam * (np.abs(w1) + np.abs(w2))
        )

        cost_history.append(cost)

        # Gradients
        dw1 = (
            (2 / n) * np.sum((y_pred - y) * X1)
            + lam * np.sign(w1)
        )

        dw2 = (
            (2 / n) * np.sum((y_pred - y) * X2)
            + lam * np.sign(w2)
        )

        db = (2 / n) * np.sum(y_pred - y)

        # Update parameters
        w1 = w1 - learning_rate * dw1
        w2 = w2 - learning_rate * dw2
        b = b - learning_rate * db

    return w1, w2, b, cost_history


# ---------------- INPUT DATA ----------------

x1_input = input("Enter X1 values: ")
x2_input = input("Enter X2 values: ")
y_input = input("Enter Y values: ")


X1 = np.array([float(v) for v in x1_input.split(",")])
X2 = np.array([float(v) for v in x2_input.split(",")])
y = np.array([float(v) for v in y_input.split(",")])


# ---------------- PARAMETERS ----------------

learning_rate = 0.01
n_iter = 25
lam = 0.5


# ---------------- TRAIN MODEL ----------------

w1, w2, b, cost_history = lasso_regression_gd(
    X1,
    X2,
    y,
    learning_rate,
    n_iter,
    lam
)


# ---------------- PRINT FINAL PARAMETERS ----------------

print(
    f"Final Parameters -> "
    f"w1: {w1:.4f}, "
    f"w2: {w2:.4f}, "
    f"b: {b:.4f}"
)


# ---------------- COST HISTORY ----------------

plt.figure(1)

plt.plot(
    range(1, n_iter + 1),
    cost_history,
    linewidth=2
)

plt.xlabel("Iteration")
plt.ylabel("Cost J(w1,w2,b)")
plt.title("Lasso Regression Cost History")

plt.grid(True)
plt.show()


# ---------------- 3D REGRESSION SURFACE ----------------

fig = plt.figure(figsize=(8, 6))

ax = fig.add_subplot(111, projection="3d")


# Original data points
ax.scatter(
    X1,
    X2,
    y,
    s=50,
    label="Data Points"
)


# Create grid
x1_grid = np.linspace(
    np.min(X1),
    np.max(X1),
    10
)

x2_grid = np.linspace(
    np.min(X2),
    np.max(X2),
    10
)

x1_grid, x2_grid = np.meshgrid(
    x1_grid,
    x2_grid
)


# Predicted Y values
y_grid = (
    w1 * x1_grid
    + w2 * x2_grid
    + b
)


# Plot regression surface
ax.plot_surface(
    x1_grid,
    x2_grid,
    y_grid,
    alpha=0.5
)


# Labels
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Y")

ax.set_title("3D Surface Fit (Lasso)")

ax.legend()

plt.show()