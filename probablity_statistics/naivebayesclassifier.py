import numpy as np
class NaiveBayes:
    def __init__(self):
        self.means = {} # Mean of features for each target value - this is cap value from slide 108?
        self.stds = {} # Standard deviation of features for each target value - this is cap value from slide 108?
        self.priors = {} # Prior probability for each target value

    def fit(self, X, y):
        """
        Train the model by calculating the mean, std, and prior probabilities for each class.
        X: Feature matrix (numpy array)
        y: Target labels (numpy array) - since this is a classifier, this will be a discrete array
        """
        self.classes = np.unique(y) # possible values of the predictions in 'y' (target labels)
        for cls in self.classes:
            X_cls = X[y == cls]
            self.means[cls] = np.mean(X_cls, axis=0)
            self.stds[cls] = np.std(X_cls, axis=0)

            # This is probability of X given y, which is essentially = n(items for current class)/total number or records
            self.priors[cls] = X_cls.shape[0] / X.shape[0]

    def _gaussian_pdf(self, x, mean, std):
        """
        Calculate the Gaussian Probability Density Function for a value x.
        This is PDF formula from slide 73
        """
        exponent = np.exp(-((x - mean) ** 2) / (2 * std ** 2))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def predict(self, X):
        """
        Predict class labels for the given data points from test data or any data likely other than train data.
        """
        predictions = []

        # For each record in test data ('X'), we will store predictions values of each distinct class in ('y') derived from train data
        for x in X:
            class_probs = {}
            for cls in self.classes:

                # Though in the theory above, log likelihood is recommended, product of each value is done here for simplicity
                likelihood = np.prod(self._gaussian_pdf(x, self.means[cls], self.stds[cls]))

                # P(y|X)  = P(X|y)           * P(y)
                posterior = self.priors[cls] * likelihood
                class_probs[cls] = posterior
            
            # Now we have likelihoods of all possible 'y', the most likely outcome/prediction for this (X) is 'y' with higher probability
            predictions.append(max(class_probs, key=class_probs.get))
        return np.array(predictions)
