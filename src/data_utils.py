"""
Data Loading and Preprocessing Utilities
========================================
Functions for loading ASCII art datasets and creating training subsets.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def load_dataset(
    image_file: str, 
    label_file: str, 
    img_height: int, 
    img_width: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load ASCII art image dataset and corresponding labels.
    
    The image file contains ASCII art where:
    - ' ' (space) = 0.0 (background)
    - '+' = 0.5 (medium intensity)
    - '#' = 1.0 (high intensity)
    
    Args:
        image_file: Path to file containing ASCII art images
        label_file: Path to file containing integer labels (one per line)
        img_height: Height of each image in characters
        img_width: Width of each image in characters
        
    Returns:
        Tuple of (images, labels) where:
        - images: np.ndarray of shape (n_samples, img_height * img_width)
        - labels: np.ndarray of shape (n_samples,)
    """
    # Load labels
    with open(label_file, 'r') as f:
        labels = np.array([int(line.strip()) for line in f if line.strip()])
    
    # Load images
    with open(image_file, 'r') as f:
        lines = [line.rstrip('\n') for line in f if line.rstrip('\n')]
    
    images = []
    num_images = len(lines) // img_height
    
    # Parse each image
    for i in range(num_images):
        current_image_lines = lines[i * img_height:(i + 1) * img_height]
        img_array = np.zeros((img_height, img_width), dtype=np.float32)
        
        for row_idx, row in enumerate(current_image_lines):
            for col_idx, char in enumerate(row[:img_width]):
                if char == '+':
                    img_array[row_idx, col_idx] = 0.5
                elif char == '#':
                    img_array[row_idx, col_idx] = 1.0
        
        images.append(img_array)
    
    # Flatten images to 1D vectors
    images = np.array(images).reshape(len(images), -1)
    
    return images, labels


def create_subset(
    X: np.ndarray, 
    y: np.ndarray, 
    percent: float,
    random_state: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a random subset of the dataset.
    
    Args:
        X: Feature array of shape (n_samples, n_features)
        y: Label array of shape (n_samples,)
        percent: Percentage of data to include (0-100)
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X_subset, y_subset)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(X)
    subset_size = int(n_samples * (percent / 100))
    
    indices = np.random.choice(n_samples, subset_size, replace=False)
    X_subset = X[indices]
    y_subset = y[indices]
    
    return X_subset, y_subset


def visualize_samples(
    images: np.ndarray, 
    labels: np.ndarray, 
    original_shape: Tuple[int, int], 
    n: int = 5,
    title: str = "Sample Images",
    figsize: Tuple[int, int] = None
):
    """
    Visualize sample images from the dataset.
    
    Args:
        images: Flattened images of shape (n_samples, height * width)
        labels: Labels of shape (n_samples,)
        original_shape: Original image shape (height, width)
        n: Number of samples to display
        title: Plot title
        figsize: Figure size (width, height)
    """
    n = min(n, len(images))
    
    if figsize is None:
        figsize = (2.5 * n, 3)
    
    plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=14, fontweight='bold')
    
    for i in range(n):
        plt.subplot(1, n, i + 1)
        img = images[i].reshape(original_shape)
        plt.imshow(img, cmap='gray', vmin=0, vmax=1)
        plt.title(f"Label: {labels[i]}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


def print_ascii_image(
    image: np.ndarray, 
    original_shape: Tuple[int, int], 
    threshold: float = 0.25
):
    """
    Print an image as ASCII art to the console.
    
    Args:
        image: Flattened image of shape (height * width,)
        original_shape: Original image shape (height, width)
        threshold: Threshold for converting to ASCII characters
    """
    img = image.reshape(original_shape)
    
    for row in img:
        line = []
        for pixel in row:
            if pixel < threshold:
                line.append(' ')
            elif pixel < 0.75:
                line.append('+')
            else:
                line.append('#')
        print(''.join(line))


def get_dataset_stats(X: np.ndarray, y: np.ndarray, name: str = "Dataset"):
    """
    Print statistics about a dataset.
    
    Args:
        X: Feature array
        y: Label array
        name: Name of the dataset
    """
    unique_labels, counts = np.unique(y, return_counts=True)
    
    print(f"\n{name} Statistics:")
    print(f"{'='*50}")
    print(f"Number of samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"Number of classes: {len(unique_labels)}")
    print(f"Class distribution:")
    
    for label, count in zip(unique_labels, counts):
        percentage = (count / len(y)) * 100
        print(f"  Class {label}: {count:4d} samples ({percentage:5.2f}%)")
    
    print(f"Feature range: [{X.min():.3f}, {X.max():.3f}]")
    print(f"{'='*50}")