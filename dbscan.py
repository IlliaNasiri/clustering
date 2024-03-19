import numpy as np
from utils import euclidean_distance

class DBSCAN:
    def __init__(self, epsilon, min_samples):
        self.epsilon = epsilon
        self.min_samples = min_samples
        pass

    def fit(self, X: np.ndarray):
        pass
