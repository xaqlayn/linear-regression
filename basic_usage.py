import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionNormal, LinearRegressionGD, mean_squared_error, r2_score

# Create sample data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Split data
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx].flatten(), y[split_idx:].flatten()

# Initialize models
model_normal = LinearRegressionNormal()
model_gd = LinearRegressionGD(learning_rate=0.1, max_iters=1000)

# Train models
model_normal.fit(X_train, y_train)
model_gd.fit(X_train, y_train)

# Evaluate
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"{model.__class__.__name__}:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Weights: {model.weights}")
    return y_pred

print("=== Evaluation Results ===")
y_pred_normal = evaluate_model(model_normal, X_test, y_test)
print()
y_pred_gd = evaluate_model(model_gd, X_test, y_test)

# Plot results
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred_normal, 'r-', label='Normal Equation', linewidth=2)
plt.title('Normal Equation')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred_gd, 'g-', label='Gradient Descent', linewidth=2)
plt.title('Gradient Descent')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()

plt.tight_layout()
plt.savefig('linear_regression_comparison.png')
plt.show()

# Plot GD loss history
if hasattr(model_gd, 'loss_history'):
    plt.figure()
    plt.plot(model_gd.loss_history)
    plt.title('Gradient Descent Loss History')
    plt.xlabel('Iterations')
    plt.ylabel('MSE')
    plt.grid(True)
    plt.savefig('gd_loss_history.png')
    plt.show()
