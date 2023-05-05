from .base_classifier import BaseClassifier
from sklearn.neural_network import MLPClassifier

class NeuralNetworkClassifier(BaseClassifier):
    def __init__(self, param_grid=None, max_iter=1500,):
        if param_grid is None:
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'activation': ['tanh', 'relu'],
                'solver': ['sgd', 'adam'],
                'alpha': [0.001, 0.01]
            }
        super().__init__(model=MLPClassifier(max_iter=max_iter), search_params=param_grid)
