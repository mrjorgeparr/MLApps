from .base_classifier import BaseClassifier
from sklearn.svm import SVC

class SVMClassifier(BaseClassifier):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 5, 10, 50],
                'kernel': ['linear', 'rbf'],
                'degree': [2, 3, 4],
            }
        super().__init__(model=SVC(), search_params=param_grid)
