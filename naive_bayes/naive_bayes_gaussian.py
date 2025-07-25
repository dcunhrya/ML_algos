import numpy as np

class NaiveBayes_Gaussian:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # Calculate mean, variance, and prior for each class
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        self._priors = np.zeros((n_classes), dtype=np.float64)

        # Calculate mean, variance, and prior for each class
        for idx, c in enumerate(self._classes):
            # Get all samples for class c 
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)
        
    def _predict(self, x):
        posteriors = []
        # Calculate posterior for each class
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior += prior
            posteriors.append(posterior)
        
        # return the class with the highest posterior probability
        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-(x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator
   
if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn import datasets

    def accruacy(y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=42)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    nb = NaiveBayes_Gaussian()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    accuracy = accruacy(y_test, predictions)
    print(f'Accuracy: {accuracy:.2f}')
