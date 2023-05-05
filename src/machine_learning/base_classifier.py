import pandas as pd 
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)

class BaseClassifier:
    def __init__(self, model, search_params, random_state=42):
        self.model = model
        self.search_params = search_params
        self.random_state = random_state
        self.rand_search = None
    


    def train(self, X_train, y_train, cv=3, metric_score='accuracy', n_iter=10):
        print(f"Starting training for classifier: {self.model.__class__.__name__}")
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

        print(f"Done with training for classifier: {self.model.__class__.__name__}")
        return result_dic

    def predict(self, X_test):
        return self.model.predict(X_test)

    def plot_confusion_matrix(self, y_test, y_pred, classifier_name, suffix):
        # Create Confustion matrix
        conf_matr = confusion_matrix(y_test, y_pred, normalize='true')

        # Plot confusion matrix
        plt.figure(figsize=(10, 10))
        sns.heatmap(conf_matr, annot=True, fmt='.2f', cmap='Blues', cbar=True)

        plt.ylabel("True Score")
        plt.xlabel("Predicted Score")
        plt.title("Confusion Matrix")

        # Create path for saving the matrix
        plot_name = suffix + "_" + classifier_name + "_confusion_matrix.png"
        save_path = os.path.join("..", "..", "figures", plot_name)

        
        # If directory doesn't exist, create it and save plot.
        os.makedirs(os.path.dirname(save_path), exist_ok=True,)
        #print(f"Saving confusion matrix to: {save_path}")
        plt.savefig(save_path)

    def map_to_four_class(self, rating):
        """
        Maps the classes to only four classes
        """
        if 1 <= rating <= 2:
            return 1
        elif 3 <= rating <= 5:
            return 2
        elif 6 <= rating <= 8:
            return 3
        elif 9 <= rating <= 10:
            return 4
        
    def map_to_two_class(self, rating):
        """
        Maps the classes to only two classes
        """
        
        if 1 <= rating <= 5:
            return 1
        elif 6 <= rating <= 10:
            return 2
        
    def evaluate_two_class(self, X_test, y_test, vectorizer_name, classifier_name):
        """
        Evaluates the trained classifier on only 2 classes
        """
        print(f"Evaluating classifier {classifier_name} on two classes.")

        y_pred = self.predict(X_test)

        
        y_test_reduced = [self.map_to_two_class(y) for y in y_test]
        y_pred_reduced = [self.map_to_two_class(y) for y in y_pred]
        self.plot_confusion_matrix(y_test_reduced, y_pred_reduced, vectorizer_name + "_two_class")

        accuracy = accuracy_score(y_test_reduced, y_pred_reduced)
        f1 = f1_score(y_test_reduced, y_pred_reduced, average='weighted')
        recall = recall_score(y_test_reduced, y_pred_reduced, average='weighted')
        precision = precision_score(y_test_reduced, y_pred_reduced, average='weighted')

        csv_results = {
        "Classifier": classifier_name,
        "Vectorizer": vectorizer_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
        }
        result_file = os.path.join("..", "..", "reports","two_classes_results.csv")
        if not os.path.isfile(result_file):
                with open(result_file, 'w', newline='') as csvfile:
                    column_names = ["Classifier", "Vectorizer", "accuracy", "f1", "precision", "recall"]
                    writer = csv.DictWriter(csvfile, fieldnames=column_names)
                    writer.writeheader()
        with open(result_file, 'a', newline='') as csvfile:
            column_names = ["Classifier", "Vectorizer", "accuracy", "f1", "precision", "recall"]
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writerow(csv_results)

        # Summarize the results for pipeline
        results = {
            "precision_weighted": precision,
            "f1_weighted": f1,
            "accuracy": accuracy,
            "recall_weighted": recall,

            "classification_report": classification_report(y_test_reduced, y_pred_reduced),
            "optimization_results": self.rand_search.cv_results_
        }
        print(f"Done with 2 class evaluation for classifier: {classifier_name}")
        return results
            
    def evaluate_four_class(self, X_test, y_test, vectorizer_name, classifier_name):
        """
        Evaluates the trained classifer on only 4 classes
        """
        print(f"Evaluating classifier {classifier_name} on 4 classes.")

        y_pred = self.predict(X_test)

        
        y_test_reduced = [self.map_to_four_class(y) for y in y_test]
        y_pred_reduced = [self.map_to_four_class(y) for y in y_pred]
        self.plot_confusion_matrix(y_test_reduced, y_pred_reduced, vectorizer_name + "_four_class")

        accuracy = accuracy_score(y_test_reduced, y_pred_reduced)
        f1 = f1_score(y_test_reduced, y_pred_reduced, average='weighted')
        recall = recall_score(y_test_reduced, y_pred_reduced, average='weighted')
        precision = precision_score(y_test_reduced, y_pred_reduced, average='weighted')

        csv_results = {
        "Classifier": classifier_name,
        "Vectorizer": vectorizer_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
        }
        result_file = os.path.join("..", "..", "reports","four_classes_results.csv")
        if not os.path.isfile(result_file):
                with open(result_file, 'w', newline='') as csvfile:
                    column_names = ["Classifier", "Vectorizer", "accuracy", "f1", "precision", "recall"]
                    writer = csv.DictWriter(csvfile, fieldnames=column_names)
                    writer.writeheader()
        with open(result_file, 'a', newline='') as csvfile:
            column_names = ["Classifier", "Vectorizer", "accuracy", "f1", "precision", "recall"]
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writerow(csv_results)

        # Summarize the results for pipeline
        results = {
            "precision_weighted": precision,
            "f1_weighted": f1,
            "accuracy": accuracy,
            "recall_weighted": recall,

            "classification_report": classification_report(y_test_reduced, y_pred_reduced),
            "optimization_results": self.rand_search.cv_results_
        }
        print(f"Done with 2 class evaluation for classifier: {classifier_name}")

        return results



    def evaluate_all_class(self, X_test, y_test, vectorizer_name, classifier_name):
        """
        Evaluates the classifier on all classes
        """
        print(f"Evaluating classifier: {classifier_name} on all classes")

        y_pred = self.predict(X_test)

        self.plot_confusion_matrix(y_test, y_pred, vectorizer_name + "_all_class")


        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')

        csv_results = {
        "Classifier": classifier_name,
        "Vectorizer": vectorizer_name,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
        }
        result_file = os.path.join("..", "..", "reports","all_classes_results.csv")
        # If the csv file does not exist. Create a new csv file with column names.
        if not os.path.isfile(result_file):
                with open(result_file, 'w', newline='') as csvfile:
                    column_names = ["Classifier", "Vectorizer", "accuracy", "f1", "precision", "recall"]
                    writer = csv.DictWriter(csvfile, fieldnames=column_names)
                    writer.writeheader()
        # Concatenates rows to the existing csv file.
        with open(result_file, 'a', newline='') as csvfile:
            column_names = ["Classifier", "Vectorizer", "accuracy", "f1", "precision", "recall"]
            writer = csv.DictWriter(csvfile, fieldnames=column_names)
            writer.writerow(csv_results)

        # Summarize the results for pipeline
        results = {
            "precision_weighted": precision,
            "f1_weighted": f1,
            "accuracy": accuracy,
            "recall_weighted": recall,

            "classification_report": classification_report(y_test, y_pred),
            "optimization_results": self.rand_search.cv_results_
        }

        print(f"Done with all class evaluation for classifier: {classifier_name}")

        return results
 