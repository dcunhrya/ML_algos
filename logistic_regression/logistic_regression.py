import numpy as np

def sigmoid(X):
    return 1/(1 + np.exp(-X))

def logistic_regression(X, w, b, threshold=0.5):
    """    
    Parameters:
    X : np.ndarray
        Input features of shape (m, n) where m is the number of samples and n is the number of features.
    w : np.ndarray
        Weights of shape (n,).
    b : float
        Bias term.
    
    Returns:
    np.ndarray
        Predicted output (0, 1) of shape (m,).
    """
    logits = X @ w + b  # Linear combination
    probabilities = sigmoid(logits)  # Apply sigmoid to get probabilities
    return (probabilities >= threshold).astype(int)

if __name__ == '__main__':
    # Example usage
    X = np.array([[1, 1], [2, 2], [-1, -1], [-2, -2]])
    w = np.array([1, 1])
    b = 0
    output = [1, 1, 0, 0]
    predictions = logistic_regression(X, w, b)
    assert np.array_equal(predictions, output), "Predictions do not match expected output"
    print("Logistic regression successful")