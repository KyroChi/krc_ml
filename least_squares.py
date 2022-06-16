import numpy as np

class _LeastSquares(object):
    def __init__(self, classifier=True):
        self.classifier = classifier
        self.bias = None
        self.beta = None

    def fit(self, X, y):
        # TODO: Check that dimensions are correct
        # TODO: Check that there are enough points
        # TODO: Check if X.T * X is singular

        # Work in place
        X = X.copy()
        y = y.copy()

        # Add column to compute bias term
        X = np.column_stack([X, np.ones(X.shape[0])])

        t1 = np.linalg.inv(X.T.dot(X))
        t1 = t1.dot(X.T)

        self.beta = t1.dot(y)[:-1]
        self.bias = t1.dot(y)[-1]

    def predict(self, X):
        if self.bias is None:
            raise Exception("Model is not trained.")

        res = self.bias + X.dot(self.beta)
        return np.where(res > 0.5, 1, 0) if self.classifier else res

class LeastSquaresClassifier(_LeastSquares):
    def __init__(self):
        super().__init__(classifier=True)


class LeastSquaresRegressor(_LeastSquares):
    def __init__(self):
        super().__init__(classifier=False)

    
if __name__ == '__main__':
    np.random.seed(42)
    
    num_samples = 10
    m = 1.5
    b = 1
    
    x1 = [ii for ii in range(num_samples)]
    x2 = [m*ii + b + np.random.normal(scale=.25) \
          for ii in range(num_samples)]

    X = np.column_stack([x1, x2])
    y = np.where(X.T[1] > m*X.T[0] + b, 1, 0)

    ls = LeastSquaresClassifier()
    ls.fit(X, y)
    Xhat = np.column_stack([[3, 4],[1, 12]])
    
    if np.any(ls.predict(Xhat) - [0, 1]) != 0:
        print("Test failed.")
    else:
        print("Test passed.")
