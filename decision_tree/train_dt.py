from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from decision_tree import DecisionTree

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the decision tree classifier
clf = DecisionTree(min_samples_split=2, max_depth=3, n_features=2)
clf.fit(X_train, y_train)

# Make predictions
predictions = clf.predict(X_test)
print("Predictions:", predictions)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.2f}%")