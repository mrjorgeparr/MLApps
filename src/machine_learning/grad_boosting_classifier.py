from .base_classifier import BaseClassifier
from sklearn.ensemble import GradientBoostingClassifier

class GradientBoostingClassifier(BaseClassifier):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [10, 50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.5, 1],
                'subsample': [0.5, 0.8, 1.0],
                'max_depth': [3, 4, 5, 6],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        super().__init__(model=GradientBoostingClassifier(), search_params=param_grid)
