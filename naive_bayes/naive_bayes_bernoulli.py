import numpy as np
from collections import Counter

class NaiveBayes_Bernoulli():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.classes = None
        self.priors = None

    def fit(self, X, y):
        self.classes, class_counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_features = X.shape[1]
        n_classes = len(self.classes)

        self.priors = np.zeros(n_classes)
        self.likelihoods = np.zeros((n_classes, n_features, 2)) 

        # Calculate likelihoods P(feature | class)
        for idx, c in enumerate(self.classes):
            X_c = X[y == c]
            class_samples = X_c.shape[0]

            # Calculate probability of feature being 1 for each feature given the class
            # np.sum(X_c, axis=0) counts the number of 1s for each feature in X_c
            # Add smoothing for both 1s and 0s in the denominator
            prob_feature_is_1 = (np.sum(X_c, axis=0) + self.smoothing) / \
                                (class_samples + 2 * self.smoothing)

            prob_feature_is_0 = 1 - prob_feature_is_1
            self.likelihoods[idx, :, 0] = np.log(prob_feature_is_0)
            self.likelihoods[idx, :, 1] = np.log(prob_feature_is_1)
            self.priors[idx] = np.log(class_counts[idx] / n_samples)

    def _compute_posterior(self, sample):
        posteriors = {}

        for idx, c in enumerate(self.classes):
            log_posterior = self.priors[idx]
            log_prob_feature_is_0 = self.likelihoods[idx, :, 0]
            log_prob_feature_is_1 = self.likelihoods[idx, :, 1]

            log_likelihood_sample_given_class = np.sum(
                sample * log_prob_feature_is_1 + (1 - sample) * log_prob_feature_is_0
            )

            # Add log-likelihood to log-prior to get log-posterior
            log_posterior += log_likelihood_sample_given_class
            posteriors[c] = log_posterior

        # We need to find the class label (c) that corresponds to the max posterior value
        return max(posteriors, key=posteriors.get)

    def predict(self, X):
        return np.array([self._compute_posterior(sample) for sample in X])

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accuracy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, n_informative=8, n_redundant=0, random_state=42)
    
    X = (X > 0.5).astype(int) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb = NaiveBayes_Bernoulli()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    accuracy_score = accuracy(y_test, predictions) # Renamed to avoid confusion with function name
    print(f'Accuracy: {accuracy_score:.2f}')

    # Test with a single sample
    sample_to_predict = X_test[0]
    single_prediction = nb._compute_posterior(sample_to_predict)
    print(f"\nPrediction for sample {sample_to_predict}: {single_prediction}")
    print(f"Actual label for this sample: {y_test[0]}")