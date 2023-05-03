import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

class BaseClassifier:
    def __init__(self, model, search_params, random_state):
        self.model = model
        self.search_params = search_params
        self.random_state = random_state
        self.rand_search = None
    


    def train(self, X_train, y_train, cv=3, metric_score='accuracy', n_iter=10):
        # Initialize the RandomizedSearch. n_iter can be adjusted for a tradeoff between quality and performance.
        self.rand_search = RandomizedSearchCV(self.model, self.search_params, cv=cv, scoring=metric_score,
                                               n_jobs=-1, n_iter=n_iter, random_state=self.random_state)
        
        self.rand_search.fit(X_train, y_train)
        self.model = self.rand_search.best_estimator_

        result_dic = {
            'best_params': self.rand_search.best_params_,
            'best_score': self.rand_search.best_score_,
            'cross_val_results': self.rand_search.cv_results_
        }

        return result_dic

    def predict(self, X_test):
        return self.model.predict(X_test)


    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)

        # Create Confustion matrix
        conf_matr = confusion_matrix(y_test, y_pred)

        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matr, annot=True, fmt='d', cmap='Blues', cbar=False)
        plt.title("Confusion Matrix")
        plt.ylabel("True Score")
        plt.xlabel("Predicted Score")
        # Create a path for saving the matrix
        plot_name = self.model.__class__.__name__  + "_confustion_matrix.png"
        save_path = os.path.join("C:", os.sep, "Users", "svenb", "Desktop", "MLApps", "figures", plot_name)

        plt.savefig(save_path)



        # Summarize evaluation results in a dictionary and return it.
        results = {
            "precision_macro": precision_score(y_test, y_pred, average='macro'),
            "f1_macro": f1_score(y_test, y_pred, average='macro'),
            "accuracy": accuracy_score(y_test, y_pred),
            "recall_macro": recall_score(y_test, y_pred, average='macro'),

            "classification_report": classification_report(y_test, y_pred),
            "optimization_results": self.rand_search.cv_results_
        }

        return results
 