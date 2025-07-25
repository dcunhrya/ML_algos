import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    # np.linalg.norm calculates the L2 norm (Euclidean distance) of the difference vector.
    # This is highly optimized and works for any number of dimensions.
    return np.linalg.norm(point1 - point2)
    # return np.sqrt(np.sum((point1 - point2) ** 2))

def get_neighbors(X_train, y_train, test_point, k):
    # Calculate distances from the test_point to all training points in a vectorized manner.
    # (X_train - test_point) uses broadcasting: test_point is effectively subtracted
    # from each row of X_train.
    # Then, **2 squares each element.
    # .sum(axis=1) sums squared differences along each row (for each training point).
    # np.sqrt takes the square root of these sums.
    distances = np.sqrt(np.sum((X_train - test_point)**2, axis=1))

    # Get the indices that would sort the distances array.
    # These indices correspond to the original positions of the training points.
    sorted_indices = np.argsort(distances)

    neighbors = []
    # Iterate through the indices of the k nearest neighbors
    # min(k, len(X_train)) ensures we don't try to get more neighbors than available training points.
    for i in range(min(k, len(X_train))):
        neighbor_idx = sorted_indices[i]
        # Append the actual training point and its label using the sorted index
        neighbors.append((X_train[neighbor_idx], y_train[neighbor_idx]))
    
    return neighbors

def predict_classification(neighbors):
    if not neighbors:
        return None

    # Extract labels from the neighbors
    neighbor_labels = [label for _, label in neighbors]

    # Count occurrences of each label
    label_counts = Counter(neighbor_labels)

    # Find the most common label
    # most_common(1) returns a list of (label, count) tuples, e.g., [('A', 3)]
    most_common_label = label_counts.most_common(1)[0][0]
    return most_common_label

# Main KNN Classifier Function (NumPy optimized inputs and internal processing)
def knn_classifier(training_data, test_data, k):
    # Convert training_data into separate NumPy arrays for features (X_train) and labels (y_train)
    # This is done once at the beginning for efficiency.
    X_train = np.array([point for point, _ in training_data], dtype=float)
    y_train = np.array([label for _, label in training_data]) # Labels can be various types

    predictions = []
    for test_point_list in test_data:
        # Convert each test point to a NumPy array just before processing
        test_point_np = np.array(test_point_list, dtype=float)
        
        # Find the k nearest neighbors for the current test point
        # Pass the NumPy-converted training data (X_train, y_train)
        neighbors = get_neighbors(X_train, y_train, test_point_np, k)
        
        # Make a prediction based on these neighbors
        prediction = predict_classification(neighbors)
        predictions.append(prediction)
        
    return predictions

# --- Example Usage ---
if __name__ == "__main__":
    # Example 1: 2D Data Classification
    print("--- Example 1: 2D Data Classification ---")
    training_data_2d = [
        ([1, 1], 'A'),
        ([1, 2], 'A'),
        ([2, 2], 'A'),
        ([5, 5], 'B'),
        ([5, 6], 'B'),
        ([6, 5], 'B'),
        ([10, 10], 'C'),
        ([10, 11], 'C'),
        ([11, 10], 'C')
    ]

    test_points_2d = [
        [1.5, 1.5], # Should be 'A'
        [5.5, 5.8], # Should be 'B'
        [9.8, 10.2], # Should be 'C'
        [3, 3]       # Ambiguous, depends on K
    ]

    k_value = 3
    predictions_2d = knn_classifier(training_data_2d, test_points_2d, k_value)

    print(f"Test points: {test_points_2d}")
    print(f"Predictions for k={k_value}: {predictions_2d}")
    # Expected: ['A', 'B', 'C', 'A'] or 'B' for [3,3] depending on tie-breaking/exact distances

    print("\n--- Example 2: 3D Data Classification ---")
    training_data_3d = [
        ([1, 1, 1], 'Red'),
        ([1, 2, 1], 'Red'),
        ([2, 1, 1], 'Red'),
        ([10, 10, 10], 'Blue'),
        ([10, 11, 10], 'Blue'),
        ([11, 10, 11], 'Blue')
    ]

    test_points_3d = [
        [1.5, 1.5, 1.5],   # Should be 'Red'
        [9.8, 10.5, 10.2]  # Should be 'Blue'
    ]

    k_value_3d = 2
    predictions_3d = knn_classifier(training_data_3d, test_points_3d, k_value_3d)

    print(f"Test points: {test_points_3d}")
    print(f"Predictions for k={k_value_3d}: {predictions_3d}")
    # Expected: ['Red', 'Blue']

    print("\n--- Example 3: Handling k > number of training points ---")
    training_data_small = [([1,1], 'X'), ([2,2], 'Y')]
    test_points_small = [[1.5, 1.5]]
    k_large = 5 # k is larger than the training data size

    predictions_small = knn_classifier(training_data_small, test_points_small, k_large)
    print(f"Test points: {test_points_small}")
    print(f"Predictions for k={k_large}: {predictions_small}")
    # Expected: ['X'] or 'Y' depending on exact distances and tie-breaking
