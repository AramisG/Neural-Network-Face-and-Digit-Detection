"""
Neural Network Implementation from Scratch
==========================================
Custom implementation of feedforward neural networks using NumPy and PyTorch tensors.
Supports multi-class classification with configurable architectures.
"""

import numpy as np
import torch
from typing import Tuple, Optional


class NeuralNetwork:
    """
    Multi-layer feedforward neural network with manual backpropagation.
    
    Architecture:
        - Input layer (size = input_dim)
        - Hidden layer 1 (size = 0.3 * input_dim, ReLU activation)
        - Hidden layer 2 (size = 0.5 * hidden1_size, ReLU activation)
        - Output layer (size = num_classes, Sigmoid activation)
    
    Args:
        input_dim: Number of input features
        num_classes: Number of output classes
        learning_rate: Learning rate for gradient descent (default: 0.7)
        batch_size: Mini-batch size for training (default: 32)
        use_torch: Whether to use PyTorch tensors instead of NumPy (default: False)
    """
    
    def __init__(
        self, 
        input_dim: int, 
        num_classes: int, 
        learning_rate: float = 0.7,
        batch_size: int = 32,
        use_torch: bool = False
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.use_torch = use_torch
        
        # Architecture dimensions
        self.hidden1_size = int(input_dim * 0.3)
        self.hidden2_size = int(self.hidden1_size * 0.5)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize weights using Xavier initialization."""
        if self.use_torch:
            dtype = torch.float64
            self.W1 = torch.randn(self.input_dim, self.hidden1_size, dtype=dtype) * \
                      torch.sqrt(torch.tensor(1.0 / self.input_dim, dtype=dtype))
            self.W2 = torch.randn(self.hidden1_size, self.hidden2_size, dtype=dtype) * \
                      torch.sqrt(torch.tensor(1.0 / self.hidden1_size, dtype=dtype))
            self.W3 = torch.randn(self.hidden2_size, self.num_classes, dtype=dtype) * \
                      torch.sqrt(torch.tensor(1.0 / self.hidden2_size, dtype=dtype))
        else:
            dtype = np.float64
            self.W1 = (np.random.randn(self.input_dim, self.hidden1_size) * \
                      np.sqrt(1.0 / self.input_dim)).astype(dtype)
            self.W2 = (np.random.randn(self.hidden1_size, self.hidden2_size) * \
                      np.sqrt(1.0 / self.hidden1_size)).astype(dtype)
            self.W3 = (np.random.randn(self.hidden2_size, self.num_classes) * \
                      np.sqrt(1.0 / self.hidden2_size)).astype(dtype)
    
    def _relu(self, x):
        """ReLU activation function."""
        return torch.relu(x) if self.use_torch else np.maximum(0, x)
    
    def _relu_derivative(self, x):
        """Derivative of ReLU activation."""
        if self.use_torch:
            return (x > 0).double()
        return (x > 0).astype(np.float64)
    
    def _sigmoid(self, x):
        """Sigmoid activation function."""
        if self.use_torch:
            return torch.sigmoid(x)
        return 1 / (1 + np.exp(-x))
    
    def _sigmoid_derivative(self, x):
        """Derivative of sigmoid activation."""
        return x * (1 - x)
    
    def _forward_pass(self, x):
        """
        Perform forward propagation through the network.
        
        Args:
            x: Input batch of shape (batch_size, input_dim)
            
        Returns:
            Tuple of (output, intermediate_activations)
        """
        # Layer 1
        f1 = x @ self.W1
        a1 = self._relu(f1)
        
        # Layer 2
        f2 = a1 @ self.W2
        a2 = self._relu(f2)
        
        # Output layer
        f3 = a2 @ self.W3
        output = self._sigmoid(f3)
        
        return output, (f1, a1, f2, a2, f3)
    
    def _backward_pass(self, x, y_onehot, output, activations):
        """
        Perform backward propagation to compute gradients.
        
        Args:
            x: Input batch
            y_onehot: One-hot encoded labels
            output: Network output from forward pass
            activations: Intermediate activations from forward pass
            
        Returns:
            Tuple of weight gradients (dW1, dW2, dW3)
        """
        f1, a1, f2, a2, f3 = activations
        
        # Output layer gradient
        d3 = (output - y_onehot) * self._sigmoid_derivative(output)
        
        # Hidden layer 2 gradient
        if self.use_torch:
            d2 = (d3 @ self.W3.T) * self._relu_derivative(f2)
        else:
            d2 = d3 @ self.W3.T * self._relu_derivative(f2)
        
        # Hidden layer 1 gradient
        if self.use_torch:
            d1 = (d2 @ self.W2.T) * self._relu_derivative(f1)
        else:
            d1 = d2 @ self.W2.T * self._relu_derivative(f1)
        
        # Compute weight gradients
        dW3 = a2.T @ d3
        dW2 = a1.T @ d2
        dW1 = x.T @ d1
        
        return dW1, dW2, dW3
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        epochs: int = 10,
        verbose: bool = True
    ) -> list:
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            X_train: Training data of shape (n_samples, input_dim)
            y_train: Training labels of shape (n_samples,)
            epochs: Number of training epochs
            verbose: Whether to print training progress
            
        Returns:
            List of average costs per epoch
        """
        if self.use_torch:
            X_train = torch.tensor(X_train, dtype=torch.float64)
            y_train = torch.tensor(y_train, dtype=torch.long)
        
        n_samples = len(X_train)
        cost_history = []
        
        # Create one-hot encoded labels
        if self.use_torch:
            y_onehot = torch.zeros(n_samples, self.num_classes, dtype=torch.float64)
            y_onehot.scatter_(1, y_train.view(-1, 1), 1)
        else:
            y_onehot = np.zeros((n_samples, self.num_classes), dtype=np.float64)
            y_onehot[np.arange(n_samples), y_train] = 1
        
        for epoch in range(epochs):
            total_cost = 0
            
            # Shuffle data
            if self.use_torch:
                indices = torch.randperm(n_samples)
            else:
                indices = np.random.permutation(n_samples)
            
            # Mini-batch training
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]
                
                x_batch = X_train[batch_idx]
                y_batch = y_onehot[batch_idx]
                
                # Forward pass
                output, activations = self._forward_pass(x_batch)
                
                # Compute loss (binary cross-entropy)
                if self.use_torch:
                    output = torch.clamp(output, 1e-15, 1 - 1e-15)
                    cost = -torch.sum(y_batch * torch.log(output) + 
                                     (1 - y_batch) * torch.log(1 - output))
                    total_cost += cost.item()
                else:
                    output = np.clip(output, 1e-15, 1 - 1e-15)
                    cost = -np.sum(y_batch * np.log(output) + 
                                  (1 - y_batch) * np.log(1 - output))
                    total_cost += cost
                
                # Backward pass
                dW1, dW2, dW3 = self._backward_pass(x_batch, y_batch, output, activations)
                
                # Update weights
                batch_size_actual = end - start
                self.W1 -= self.learning_rate * (dW1 / batch_size_actual)
                self.W2 -= self.learning_rate * (dW2 / batch_size_actual)
                self.W3 -= self.learning_rate * (dW3 / batch_size_actual)
            
            avg_cost = total_cost / n_samples
            cost_history.append(avg_cost)
            
            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_cost:.4f}")
        
        return cost_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on input data.
        
        Args:
            X: Input data of shape (n_samples, input_dim)
            
        Returns:
            Predicted class labels of shape (n_samples,)
        """
        if self.use_torch:
            X = torch.tensor(X, dtype=torch.float64)
        
        output, _ = self._forward_pass(X)
        
        if self.use_torch:
            predictions = torch.argmax(output, dim=1).numpy()
        else:
            predictions = np.argmax(output, axis=1)
        
        return predictions
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> float:
        """
        Evaluate model accuracy on test data.
        
        Args:
            X_test: Test data of shape (n_samples, input_dim)
            y_test: Test labels of shape (n_samples,)
            
        Returns:
            Accuracy percentage (0-100)
        """
        predictions = self.predict(X_test)
        
        if self.use_torch and isinstance(y_test, torch.Tensor):
            y_test = y_test.numpy()
        
        accuracy = np.mean(predictions == y_test) * 100
        return accuracy
    
    def get_weights(self) -> Tuple:
        """Return current weight matrices."""
        return (self.W1, self.W2, self.W3)