import numpy as np
from pandas import DataFrame
from tqdm import tqdm
from models.guides import Model


class LinearRegressionGD(Model):
    """
    Linear Regression implemented from scratch using Gradient Descent.

    Instead of the closed-form normal equation w = (X^T X)^-1 X^T y,
    this implementation iteratively minimizes MSE via gradient descent.
    The bias term is folded into the weight vector by prepending a column
    of ones to X, so no separate bias variable is needed.

    Parameters
    ----------
    lr : float
        Learning rate (step size for each gradient update). Default 0.01.
    n_iter : int
        Maximum number of gradient descent iterations. Default 1000.
    tol : float
        Early stopping tolerance. Training halts when the change in MSE
        between iterations falls below this threshold. Default 1e-6.

    Attributes
    ----------
    weights : np.ndarray
        Learned weight vector after fitting. weights[0] is the bias term,
        weights[1:] are the feature coefficients.
    loss_history : list
        MSE recorded at each iteration. Useful for plotting convergence.
    """

    def __init__(self, lr: float = 0.01, n_iter: int = 1000, tol: float = 1e-6):
        self.lr = lr
        self.n_iter = n_iter
        self.tol = tol
        self.weights = None
        self.loss_history = []  # Store loss per iteration for convergence plots

    def fit(self, X: DataFrame, y: DataFrame) -> None:
        """
        Train the model using gradient descent.

        The update rule is derived from the MSE loss:
            L(w) = (1/n) * ||y - Xw||^2
            grad  = -(2/n) * X^T (y - y_hat)
            w    -= lr * grad

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features. Must be normalized before passing in
            for gradient descent to converge reliably.
        y : array-like of shape (n_samples,)
            Training target values.
        """
        # Convert X and y into numpy arrays
        X = np.array(X)
        y = np.array(y)

        # Prepend a ones column to X so the bias is handled as weights[0]
        X = np.column_stack([np.ones(X.shape[0]), X])

        # Initialize weight vector to zeros — shape matches features + bias
        weights = np.zeros(X.shape[1])

        n = y.shape[0]  # Number of samples
        self.loss_history = []  # Reset loss history for multiple fit calls

        for i in tqdm(range(self.n_iter), desc="Fitting Model"):
            # Forward pass — compute predictions via matrix multiplication
            yi = X @ weights

            # Compute MSE loss
            mse = np.mean(np.square(y - yi))

            # Compute gradient: -(2/n) * X^T (y - y_hat)
            grad = -(2 / n) * X.T @ (y - yi)

            # Gradient descent update — step opposite to gradient (downhill)
            weights -= self.lr * grad

            # Record loss for convergence analysis
            self.loss_history.append(mse)

            # Early stopping — halt if improvement is below tolerance
            if len(self.loss_history) > 1 and abs(self.loss_history[-2] - mse) < self.tol:
                tqdm.write(f"Early stopping at iteration {i} — loss change below {self.tol}")
                break

        self.set_weights(weights)

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        Generate predictions for input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Must contain the same features in the same order as fit().

        Returns
        -------
        np.ndarray of shape (n_samples,)
        """
        # Convert to numpy and prepend bias column — same as fit()
        X = np.array(X)
        X = np.column_stack([np.ones(X.shape[0]), X])
        return X @ self.get_weights()

    def score(self, X: DataFrame, y: DataFrame) -> float:
        """
        Compute R-squared — proportion of variance in y explained by the model.

        R² = 1 - (SS_res / SS_tot)
            SS_res = sum of squared residuals (model error)
            SS_tot = total variance in y (baseline error — predicting the mean)

        R² = 1.0 → perfect fit
        R² = 0.0 → no better than predicting the mean
        R² < 0.0 → worse than predicting the mean

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        float : R² score
        """
        y = np.array(y)
        yi = self.predict(X)

        # Residual sum of squares — model error
        ss_res = np.sum(np.square(y - yi))

        # Total sum of squares — variance of y around its mean (null baseline)
        ss_tot = np.sum(np.square(y - np.mean(y)))

        return 1 - (ss_res / ss_tot)

    def get_coeff(self) -> dict:
        """
        Return the learned intercept and feature coefficients as a dictionary.

        weights[0] is the bias/intercept (from the prepended ones column).
        weights[1:] are the coefficients for each input feature.
        """
        return {
            "intercept": self.get_weights()[0],
            "coefficients": self.get_weights()[1:]
        }

    def set_weights(self, weights: np.ndarray) -> None:
        """Store trained weights to the instance after fitting."""
        self.weights = weights

    def get_weights(self) -> np.ndarray:
        """Return the current weight vector."""
        return self.weights
