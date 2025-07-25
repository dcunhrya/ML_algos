import numpy as np

# OLS
def OLS(X, y):
    #(XtX)^-1 (Xty)
    # May need to add a column of 1s if no bias term
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    return XtX_inv @ X.T @ y

# Gradient descent
def grad_des(X, y, iterations=1000, lr=0.01):
    # Start with theta of 0s as a guess
    m, n = X.shape
    theta = np.zeros((n,1))
    for _ in range(iterations):
        # Calculate predictions
        preds = X @ theta #shape mx1
        # Calculate MSE
        error = (preds - y.reshape(-1, 1)) / m # shape mx1
        # Apply MSE error to features
        update = X.T @ error # shape nx1
        theta -= lr*update
    return theta

def testing_func(test_type, X, y, OLS_true, grad_true, iterations=1000, lr=0.01):
    if test_type == 'OLS':
        ans = OLS(X, y)
        np.testing.assert_array_almost_equal(ans, OLS_true, decimal=4)
        print('OLS success')
    else:
        ans = grad_des(X, y, iterations, lr)
        np.testing.assert_array_almost_equal(ans, grad_true, decimal=4)
        print('Gradient descent success')


if __name__ == '__main__':
    X = np.array([[1, 1], [1, 2], [1, 3]])
    y = np.array([1,2,3])
    OLS_true = np.array([0.0, 1.0])
    grad_true = np.array([[0.1107], [0.9513]])
    test_type = 'OLS'
    # test_type = 'grad_des'
    testing_func(test_type, X, y, OLS_true, grad_true)
