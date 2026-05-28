import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.decomposition import PCA
from kneed import KneeLocator
import statsmodels.api as sm

class OLS:
    def __init__(self):
        self.model = None


    def fit(self, X, Y):
        # find weights in the form of a 1D numpy array such that, for each stock, multiply them by feature_count weights 
        # where each column has its own weight. Thus, there will be a feature_count amount of weights - and a row_number 
        # amount of multiplications for each weight. The sum of all of these weights and values in the columns
        # is being best adjusted to equal the return. These weights are the closest possible fit - a minimization of the sum of
        # squared errors between X @ w and Y. This is OLS.
        # the actual math: self.weights = np.linalg.inv(X.T @ X) @ X.T @ Y
        X_constant = sm.add_constant(X)
        self.model = sm.OLS(Y, X_constant).fit()
        # the stable version ^

    def predict(self, X):
        if self.weights is None:
            raise ValueError("Model not fitted yet. Call fit(input: X, target: Y) first.")
        # multiply input df X by the weights such that Xw = Y where Y represents predictions
        predictions = self.model.predict(X)
        return predictions
    

class ElasticNetModel:
    def __init__(self, alpha=0.001, l1_ratio=0.5):
        # note that elastic net is very similar to OLS, just with penalization - alpha signifies how strong penalization is, 
        # l1_ratio is the different types alpha "acts through", that is, if l1_ratio = 1, it's all l1 - and this zeroes out
        # large features, or l2, which shrinks them, which is represented by l1_ratio = 0. These are the hyperparameters,
        # and this is the elastic net.
        self.model = None
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.model = None
    
    # buffers the model using the ElasticNet class as provided by scikit-learn. 
    def fit(self, X, Y):
        self.model = ElasticNet(self.alpha, self.l1_ratio)
        self.model.fit(X, Y)
    
    def predict(self, X):
        # provides a predictions numpy array, given any data frame X such that X represents features/input to the model
        if self.model is None:
            raise ValueError("Model not fitted yet. Call fit(input: X, target: Y) first.")
        predictions = self.model.predict(X)
        return predictions
    
class PCAModel:
    # basically OLS with component grouping
    def __init__(self, n_components):
        self.n_components = n_components
        self.pca = None
        self.model = None

    def optimal_components(self, X_train):
        pca = PCA()

        pca.fit(X_train)
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        max_pcs = min(20, X_train.shape[1])
        pcs = np.arange(1, max_pcs + 1)

        explained_variance = explained_variance[:max_pcs]

        kneedle = KneeLocator(pcs, explained_variance, curve="convex", direction="decreasing")
        knee_val = kneedle.knee

        return knee_val
    
    def fit(self, X, Y):
        self.n_components = self.optimal_components(X)

        self.pca = PCA(n_components=self.n_components)
        X_reduced = sm.add_constant(self.pca.fit_transform(X))

        self.model = sm.OLS(Y, X_reduced).fit()
        

    def predict(self, X):
        X_reduced = sm.add_constant(self.pca.transform(X))
        pred = self.model.predict(X_reduced)
        
        return pred
    

    
    