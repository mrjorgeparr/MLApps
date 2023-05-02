from .base_classifier import BaseClassifier
from sklearn.neural_network import MLPClassifier

class NeuralNetworkClassifier(BaseClassifier):
    def __init__(self, param_grid=None):
        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['logistic', 'tanh', 'relu'],
                'solver': ['lbfgs', 'sgd', 'adam'],
                'alpha': [0.0001, 0.001, 0.01, 0.1]
            }
        super().__init__(model=MLPClassifier(), search_params=param_grid)
