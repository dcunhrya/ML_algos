
import numpy as np

def euclidean_distance(a, b):
    return np.sqrt(((a - b) ** 2).sum(axis=1))

def k_means_clustering(points, k, initial_centroids, max_iterations):
    points = np.array(points)
    centroids = np.array(initial_centroids)
    
    for iteration in range(max_iterations):
        # Assign points to the nearest centroid
        # distances shape: (k, n_points)
        distances = np.array([euclidean_distance(points, centroid) for centroid in centroids])
        # assignments shape: (n_points,)
        assignments = np.argmin(distances, axis=0)

        # Calculate new centroids
        new_centroids = np.copy(centroids)
        for i in range(k):
            assigned_points = points[assignments == i]
            if len(assigned_points) > 0:
                new_centroids[i] = assigned_points.mean(axis=0)
        
        # Check for convergence
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
        centroids = np.round(centroids,4)
    return [tuple(centroid) for centroid in centroids]

if __name__ == '__main__':
    points = [(1, 2), (1, 4), (1, 0), (10, 2), (10, 4), (10, 0)]
    k = 2
    initial_centroids = [(1, 1), (10, 1)]
    max_iterations = 10
    true_centroids = [(1, 2), (10, 2)]
    centroids = k_means_clustering(points, k, initial_centroids, max_iterations)
    assert centroids == true_centroids, f"Expected {true_centroids}, but got {centroids}"
    print('K-means clustering success!')
