import numpy as np

class LinearRegressionNormal:
    """Linear Regression using Normal Equation (closed-form solution)"""
    
    def __init__(self, fit_intercept=True):
        self.fit_intercept = fit_intercept
        self.weights = None
    
    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Validate shapes
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
            
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        try:
            XT = X.T
            self.weights = np.linalg.inv(XT @ X) @ XT @ y
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if matrix is singular
            self.weights = np.linalg.pinv(X) @ y
    
    def predict(self, X):
        # Convert to numpy array
        X = np.array(X)
        
        # Handle 1D input
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        return X @ self.weights


class LinearRegressionGD:
    """Linear Regression using Gradient Descent"""
    
    def __init__(self, fit_intercept=True, learning_rate=0.01, max_iters=1000, tol=1e-4):
        self.fit_intercept = fit_intercept
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.tol = tol
        self.weights = None
        self.loss_history = []
    
    def fit(self, X, y):
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Validate shapes
        if len(X.shape) == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have the same number of samples")
            
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        
        for epoch in range(self.max_iters):
            y_pred = X @ self.weights
            error = y_pred - y
            mse = np.mean(error**2)
            self.loss_history.append(mse)
            
            gradient = (2/n_samples) * (X.T @ error)
            new_weights = self.weights - self.learning_rate * gradient
            
            # Check convergence
            weight_change = np.linalg.norm(new_weights - self.weights)
            if weight_change < self.tol:
                break
                
            self.weights = new_weights
    
    def predict(self, X):
        # Convert to numpy array
        X = np.array(X)
        
        # Handle 1D input
        if len(X.shape) == 1:
            X = X.reshape(1, -1)
            
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
            
        return X @ self.weights
