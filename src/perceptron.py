"""
Perceptron Classifier Implementation
Multi-class perceptron using the one-vs-all approach.
"""

import numpy as np
from typing import Optional


class Perceptron:
    """
    Multi-class perceptron classifier.
    
    Uses the perceptron learning algorithm with one weight vector per class.
    Updates weights when a prediction is incorrect by moving the correct
    class weights toward the input and the predicted class weights away.
    
    Args:
        num_classes: Number of output classes
        learning_rate: Learning rate for weight updates (default: 0.05)
        max_epochs: Maximum number of training epochs (default: 10)
        random_state: Random seed for reproducibility (default: None)
    """
    
    def __init__(
        self,
        num_classes: int,
        learning_rate: float = 0.05,
        max_epochs: int = 10,
        random_state: Optional[int] = None
    ):
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.random_state = random_state
        self.weights = None
        self.training_errors = []
        
    def _initialize_weights(self, n_features: int):
        """Initialize weight matrix with random values."""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        self.weights = np.random.rand(self.num_classes, n_features)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True):
        """
        Train the perceptron using the perceptron learning algorithm.
        
        The algorithm iterates through training examples and updates weights
        when predictions are incorrect:
        - Increase weights for the correct class (move toward input)
        - Decrease weights for the predicted class (move away from input)
        
        Args:
            X_train: Training data of shape (n_samples, n_features)
            y_train: Training labels of shape (n_samples,)
            verbose: Whether to print training progress
            
        Returns:
            self for method chaining
        """
        n_samples, n_features = X_train.shape
        self._initialize_weights(n_features)
        self.training_errors = []
        
        for epoch in range(self.max_epochs):
            mistakes = 0
            
            #Iterate through each training example
            for i in range(n_samples):
                x = X_train[i]
                y_true = y_train[i]
                
                #Compute scores for all classes
                scores = self.weights @ x
                y_pred = np.argmax(scores)
                
                #Update weights if prediction is wrong
                if y_pred != y_true:
                    #Move correct class weights toward input
                    self.weights[y_true] += self.learning_rate * x
                    #Move predicted class weights away from input
                    self.weights[y_pred] -= self.learning_rate * x
                    mistakes += 1
            
            error_rate = mistakes / n_samples
            self.training_errors.append(error_rate)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{self.max_epochs} - "
                      f"Errors: {mistakes}/{n_samples} ({error_rate:.2%})")
            
            if mistakes == 0:
                if verbose:
                    print(f"Converged at epoch {epoch + 1}")
                break
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for input data.
        
        Args:
            X: Input data of shape (n_samples, n_features)
            
        Returns:
            Predicted labels of shape (n_samples,)
        """
        if self.weights is None:
            raise ValueError("Model must be trained before making predictions")
        
        scores = X @ self.weights.T
        predictions = np.argmax(scores, axis=1)
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate model accuracy on test data.
        
        Args:
            X_test: Test data of shape (n_samples, n_features)
            y_test: Test labels of shape (n_samples,)
            
        Returns:
            Accuracy percentage (0-100)
        """
        predictions = self.predict(X_test)
        accuracy = np.mean(predictions == y_test) * 100
        return accuracy
    
    def get_weights(self) -> np.ndarray:
        """Return the weight matrix."""
        return self.weights.copy() if self.weights is not None else None