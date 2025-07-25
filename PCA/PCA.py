import numpy as np

#k is number of components
def PCA(matrix, k):
    # standardize and center
    matrix_standard = (matrix - np.mean(matrix, axis=0)) / np.std(matrix, axis=0)
    cov_matrix = np.cov(matrix_standard, rowvar=False)
    # find eigenvalues
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    # return top components
    # Sort eigenvalues in descending order and get corresponding eigenvectors
    idx = np.argsort(eigenvalues)[::-1]
    eig_val_sorted = eigenvalues[idx]
    eig_vec_sorted = eigenvectors[:,idx]

    return np.round(eig_vec_sorted[:, :k], 4)

if __name__ == "__main__":
    x = np.array([[1, 2], [3, 4], [5, 6]])
    k = 1
    y = [[0.7071], [0.7071]]
    prediction = PCA(x, k)
    np.testing.assert_array_almost_equal(prediction, y, decimal=4)
    print('Test passed!')
