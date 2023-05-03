from .base_classifier import BaseClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np

class LogisticRegressionClassifier(BaseClassifier):
    def __init__(self, max_iter=1000, random_state=42):
        classifier = LogisticRegression(max_iter=max_iter, multi_class='auto')
        param_grid = {
            'C': np.logspace(-4, 4, 20),
            'penalty': ['l2', 'l2',],
            'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']
        }
        super().__init__(classifier, param_grid, random_state)