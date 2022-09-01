import numpy as np
import statistics

def _metric_euclidian(x, y):
    return np.sqrt( (x-y).dot(x-y) )

def _metric_l1(x, y):
    return np.abs( x - y )


class _KNearestNeighbor(object):
    """
    k-Nearest Neighbor Model
    
    This is a base class. Use KNearestNeighborClassifier or
    KNearestNeighborRegressor for implemenations
    
    Hashed-Based lookup
    
    Attributes
    ==========
    k: odd integer
        The number of neighbors to look up
    
    Methods
    =======
    fit(X, y)

    predict(X)

    _predict(x)
    """
    def __init__(self, k, metric='euclidian', classifier=True):
        self.k = k
        
        # Choose an odd k value to avoid disagreements
        if self.k % 2 == 0:
            raise Exception("k must be an odd number")
        
        self.Xy = None
        self.input_size = None
        self.sample_size = None
        
        self.classifier = classifier
        
        if metric == 'euclidian':
            self.metric = _metric_euclidian
        elif metric == 'l1':
            self.metric = _metric_l1
        else:
            raise Exception("Unknown metric: %s" % metric)

    def fit(self, X, y):
        # TODO: Check inputs
        
        self.input_size, self.sample_size = X.shape
        self.Xy = np.column_stack([X, y])

    def predict(self, X):
        self.check_none(self.Xy, "Model has not been trained.")
        
        yhat = []
        for x in X:
            yhat.append(self._predict(x))
        return yhat

    def _predict(self, x):
        # Need a way to handle multiple points with the same distance

        neighbors = {}

        for pt in self.Xy:
            dist = self.metric(pt[:-1], x)
            if dist in neighbors.keys():
                neighbors[dist].append(pt[-1])
            else:
                neighbors[dist] = [pt[-1]]
                
        neighbors_left = self.k
        neighbor_vals = []

        keys = list(neighbors.keys())
        keys.sort()

        for dist in keys:
            if neighbors_left > 0:
                neighbor_vals.append(*neighbors[dist])
                neighbors_left -= len(neighbors[dist])
            else:
                break

        # If there are multiple values with same distance this
        # distance gets weighted more heavily.
        val = statistics.median(neighbor_vals)
#         val = sum(neighbor_vals) / len(neighbor_vals)
        if self.classifier: 
            return round(val) 
        else: 
            return val

    def get_neighbors(self, x):
        self.check_none(self.Xy, "Model has not been trained.")

    def check_none(self, obj, msg):
        if obj is None:
            raise Exception(msg)


class _KNearestNeighborKDTree(object):
    """ Implementation of k nearest neighbors using a k-d tree instead
    of simply iterating through the points. Should run in O(log n)
    instead of O(n) """
    def __init__(self, k, metric='euclidian'):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def _predict(self, x):
        pass
    
class KNearestNeighborClassifier(_KNearestNeighbor):
    def __init__(self, k, **kwargs):
        super().__init__(k, **kwargs, classifier=True)


class KNearestNeighborRegressor(_KNearestNeighbor):
    def __init__(self, k, **kwargs):
        super().__init__(k, **kwargs, classifier=False)


if __name__ == '__main__':
    np.random.seed(42)
    
    knn = _KNearestNeighbor(10)

    test_pts = [
        [1, 1], [1, 2], [.5, .25], [.5, 1]
    ]

    num_sample_pts = 30
    normal_scale=0.3

    points = []
    for pt in test_pts:
        for ii in range(num_sample_pts):
            points.append(
                [pt[0] + np.random.normal(scale=normal_scale),
                 pt[1] + np.random.normal(scale=normal_scale)]
            )

    knn.fit
