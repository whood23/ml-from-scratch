from abc import ABC, abstractmethod


class Model(ABC):
    """
    Abstract base class for all from-scratch model implementations.

    Enforces a consistent interface across all supervised models:
    every model must implement fit, predict, and score.
    This mirrors the sklearn estimator API convention.
    """

    @abstractmethod
    def fit(self, X, y):
        """Train the model on features X and target y."""
        pass

    @abstractmethod
    def predict(self, X):
        """Generate predictions for input X."""
        pass

    @abstractmethod
    def score(self, X, y):
        """Evaluate model performance on X against ground truth y."""
        pass
