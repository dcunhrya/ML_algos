import numpy as np
from collections import Counter

class Node():
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None
    
class DecisionTree():
    def __init__(self, min_samples_split, max_depth, n_features):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # Check stopping criteria
        if (depth >= self.max_depth or n_labels==1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # Find best split
        best_threshold, best_feature = self._best_split(X, y, feat_idxs)

        # Create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_threshold)
        left_child = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_child = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left_child, right=right_child)

    def _most_common_label(self, y):
        counter = Counter(y)
        # most_common(1) returns a list of (label, count) tuples, e.g., [('A', 3)]
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _best_split(self, X, y, feat_idx):
        best_gain = -1
        split_threshold, split_idx = None, None

        for idx in feat_idx:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(X_column, y, idx, threshold)
            
                if gain > best_gain:
                    best_gain = gain
                    split_threshold = threshold
                    split_idx = idx

        return split_threshold, split_idx
    
    def _information_gain(self, X_col, y, feature_idx, threshold):
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idx, right_idx = self._split(X_col, threshold)
        if len(left_idx) == 0 or len(right_idx) == 0:
            return 0
        
        # calculate entropy of children
        n = len(y)
        n_l, n_r = len(left_idx), len(right_idx)
        e_l, e_r = self._entropy(y[left_idx]), self._entropy(y[right_idx])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate information gain
        gain = parent_entropy - child_entropy
        return gain

    def _split(self, X_col, threshold):
        left_idx = np.argwhere(X_col <= threshold).flatten()
        right_idx = np.argwhere(X_col > threshold).flatten()
        return left_idx, right_idx
    
    def _entropy(self, y):
        hist = np.bincount(y)
        p = hist / len(y)
        entropy = -np.sum(p * np.log2(p + 1e-9))  # avoid log(0)
        return entropy
            
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        feature_val = x[node.feature]
        if feature_val <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

