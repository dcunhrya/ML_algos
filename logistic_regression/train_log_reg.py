import numpy as np

def sigmoid(X):
    return 1 / (1+np.exp(-X))

def BCE(probabilities, y):
    epsilon = 1e-10
    y = y.reshape(probabilities.shape)
    loss = -(1-y)*(np.log(1-probabilities+epsilon)) - y*np.log(probabilities+epsilon)    
    return np.sum(loss)

def gradient_weight(X, probabilities, y):
    y = y.reshape(probabilities.shape)
    return X.T @ (probabilities - y)

def gradient_bias(probabilities, y):
    y = y.reshape(probabilities.shape)
    return np.sum(probabilities - y)

def train_logreg(X, y, lr=0.001, epochs=100) -> tuple[list[float], ...]:
    total_loss = []
    weights = np.random.randn(X.shape[1], 1) * 0.01
    bias = np.zeros((1,1))
    y = y.reshape(-1, 1)
    for _ in range(epochs):
        logits = X @ weights + bias
        probabilities = sigmoid(logits)
        loss = BCE(probabilities, y)
        total_loss.append(loss)
        weight_update = gradient_weight(X, probabilities, y)
        weights -= lr * weight_update
        bias_update = gradient_bias(probabilities, y)
        bias -= lr  * bias_update
    coefficients = [round(bias.item(), 4)] + [round(w, 4) for w in weights.flatten()]
    return coefficients, np.round(total_loss, 4)

if __name__ == '__main__':
    X = np.array([[0.7674, -0.2341, -0.2341, 1.5792], [-1.4123, 0.3142, -1.0128, -0.9080],
                   [-0.4657, 0.5425, -0.4694, -0.4634], [-0.5622, -1.9132, 0.2419, -1.7249],
                     [-1.4247, -0.2257, 1.4656, 0.0675], [1.8522, -0.2916, -0.6006, -0.6017],
                       [0.3756, 0.1109, -0.5443, -1.1509], [0.1968, -1.9596, 0.2088, -1.3281],
                         [1.5230, -0.1382, 0.4967, 0.6476], [-1.2208, -1.0577, -0.0134, 0.8225]])
    y =  np.array([1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
    truth = [-0.0701, 0.2456, 0.122, 0.1322, 0.2664]
    coefficients, _ = train_logreg(X, y)
    assert np.allclose(coefficients, truth, atol=1e-1), "Coefficients do not match expected output"
    # assert np.allclose(loss, truth[1], atol=1e-4), "Loss does not match expected output"