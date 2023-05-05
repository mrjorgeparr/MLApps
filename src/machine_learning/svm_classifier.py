from .base_classifier import BaseClassifier
from sklearn.svm import SVC

class SVMClassifier(BaseClassifier):
    def __init__(self, param_grid=None, class_weight=None):
        if param_grid is None:
            param_grid = {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf'],
                'degree': [2, 3, 4],
            }
        super().__init__(model=SVC(class_weight=class_weight), search_params=param_grid, )
