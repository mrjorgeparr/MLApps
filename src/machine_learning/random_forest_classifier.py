from .base_classifier import BaseClassifier
from sklearn.ensemble import RandomForestClassifier as RFC

class RandomForestClassifier(BaseClassifier):
    def __init__(self, param_grid=None, class_weight=None):
        if param_grid is None:
            param_grid = {
                'n_estimators': [100, 200],  
                'criterion': ['gini', 'entropy'],  
                'max_depth': [None, 20, 30],  
                'min_samples_split': [2, 5],  
                'min_samples_leaf': [1, 2]  
            }
        super().__init__(model=RFC(class_weight=class_weight), search_params=param_grid, )
