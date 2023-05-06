from .base_classifier import BaseClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, max_iter=5000, random_state=42, class_weight=None, n_iter=50):
        classifier = LogisticRegression(max_iter=max_iter, multi_class='auto', class_weight=class_weight)
        param_grid = {
            'C': np.logspace(-3, 3, 7),
            'penalty': ['l1', 'l2', 'elasticnet'],
            'solver': ['liblinear', 'saga'],
            'l1_ratio': np.linspace(0, 1, 6)  # only used when penalty='elasticnet'
        }
        super().__init__(classifier, param_grid, random_state, n_iter=n_iter)