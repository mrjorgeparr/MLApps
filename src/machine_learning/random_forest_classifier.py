from .base_classifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier

class RandomForestClassifier(BaseClassifier):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [10, 50, 100, 200],
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        super().__init__(model=RandomForestClassifier(), search_params=param_grid)
