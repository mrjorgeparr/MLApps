from .base_classifier import BaseClassifier
from sklearn.ensemble import GradientBoostingClassifier as SKBC

class GradientBoostingClassifier(BaseClassifier):
    def __init__(self, param_grid=None, random_state=42, class_weight=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [50, 100],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0],
                'max_depth': [3, 4],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2]
            }
        super().__init__(model=SKBC(), search_params=param_grid, random_state=random_state)
