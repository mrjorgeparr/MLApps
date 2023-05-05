from .base_classifier import BaseClassifier
from sklearn.neighbors import KNeighborsClassifier

class KNNClassifier(BaseClassifier):
    def __init__(self, param_grid=None, random_state=42, ):
        if param_grid is None:
            param_grid = {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'algorithm': ['auto', 'brute']
            }
        super().__init__(model=KNeighborsClassifier(), search_params=param_grid, random_state=random_state, )
