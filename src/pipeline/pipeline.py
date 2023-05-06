import os
import sys
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.text_preprocesser import TextPreprocessor
from vectorization.bow_idf_vectorizer import BoWVectorizer
from vectorization.word2vec_vectorizer import Word2VecVectorizer

from machine_learning.logistic_regression_classifier import LogisticRegressionClassifier
from machine_learning.knn_classifier import KNNClassifier
from machine_learning.grad_boosting_classifier import GradientBoostingClassifier
from machine_learning.neural_network_classifier import NeuralNetworkClassifier
from machine_learning.random_forest_classifier import RandomForestClassifier
from machine_learning.svm_classifier import SVMClassifier

from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, raw_data_path, preprocess_data_path, load_preprocessed=False, 
                 preprocess_text_column="cleaned_text", preprocess_label_column= "labels"):
        self.raw_data_path = raw_data_path
        self.processed_data_path = preprocess_data_path
        self.load_preprocessed = load_preprocessed
        self.preprocess_text_column = preprocess_text_column
        self.preprocess_label_column = preprocess_label_column
        self.labels = None
        self.preprocessed_text = None

        # Initialize components
        self.preprocessor = TextPreprocessor(self.raw_data_path)
        self.vectorizers = {
            # "bow_count": BoWVectorizer(filename= self.processed_data_path,
            #                            text_column=self.preprocess_text_column,
            #                            method="count"),
            # "bow_tfidf": BoWVectorizer(filename= self.processed_data_path,
            #                            text_column=self.preprocess_text_column,
            #                            method="tfidf"),
            # "word2vec_1gram": Word2VecVectorizer(filename= self.processed_data_path,
            #                                text_column=self.preprocess_text_column,
            #                                n=1),
            # "word2vec_3gram": Word2VecVectorizer(filename= self.processed_data_path,
            #                                 text_column=self.preprocess_text_column,
            #                                 n=3),
            # "word2vec_1gram_w8": Word2VecVectorizer(filename= self.processed_data_path,
            #                                text_column=self.preprocess_text_column,
            #                                n=1, window=8),
            "word2vec_5gram": Word2VecVectorizer(filename= self.processed_data_path,
                                            text_column=self.preprocess_text_column,
                                            n=5),

        }
        self.classifiers = {"Logistic Regression": LogisticRegressionClassifier(random_state=42, class_weight=None),
                            # "KNN": KNNClassifier(random_state=42),
                            # "GradientBoosting": GradientBoostingClassifier(random_state=42),
                            # "NeuralNetwork": NeuralNetworkClassifier(),
                            # "RandomForest": RandomForestClassifier(class_weight=None),
                            # "SVM": SVMClassifier(class_weight=None),
                            # "Logistic Regression_balance": LogisticRegressionClassifier(random_state=42, class_weight='balanced'),
                            # "RandomForest_balance": RandomForestClassifier(class_weight='balanced'),
                            # "SVM_balance": SVMClassifier(class_weight='balanced')
                            }  # More classifiers can be added here

    def add_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def add_vectorizer(self, vectorizer_name, vectorizer):
        self.vectorizers[vectorizer_name] = vectorizer

    def add_classifier(self, classifier_name, classifier):
        self.classifiers[classifier_name] = classifier

    def preprocess(self):

        output_filename = os.path.join(self.processed_data_path, 'preprocessed_data.csv')
        self.preprocessor.preprocess_labels(label_column="rating")
        self.preprocessed_text, self.labels = self.preprocessor.preprocess_corpus(output_filename)


    def execute(self):
        if self.load_preprocessed:
            temp_df = pd.read_csv(self.processed_data_path)
            self.preprocessed_text = temp_df[self.preprocess_text_column]
            self.labels = temp_df[self.preprocess_label_column]
        else:
            self.preprocess()
        print(f"Length of preprocessed_data: {len(self.preprocessed_text)}")
        print(f"Length of labels: {len(self.labels)}")

        # Vectorize the data
        for vectorizer_name, vectorizer in self.vectorizers.items():
            vectorized_data = vectorizer.fit_transform()

            if isinstance(vectorized_data, np.ndarray):
                print(f"Length of vectorized_data ({vectorizer_name}): {len(vectorized_data)}")
            else:
                print(f"Length of vectorized_data ({vectorizer_name}): {vectorized_data.shape[0]}")
            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(vectorized_data, self.labels, test_size=0.2, random_state=42)

            # Train and evaluate classifiers
            for classifier_name, classifier in self.classifiers.items():
                # Train the classifier
                classifier.train(X_train, y_train)

                # Evaluate the classifier on all classes and save results.
                eval_results_all_classes = classifier.evaluate_all_class(X_test, y_test, vectorizer_name, classifier_name)
                self.save_results(vectorizer_name, classifier_name, eval_results_all_classes)

                # Evaluate classifier on 4 classes and save results.
                eval_results_four_classes = classifier.evaluate_four_class(X_test, y_test, vectorizer_name, classifier_name)
                self.save_results(vectorizer_name, classifier_name, eval_results_four_classes)

                # Evaluate classifier on 2 classes and save results.
                eval_results_two_classes = classifier.evaluate_two_class(X_test, y_test, vectorizer_name, classifier_name)
                self.save_results(vectorizer_name, classifier_name, eval_results_two_classes)


    def save_results(self, vectorizer_name, classifier_name, eval_results):
        # Define a directory for saving the results
        results_dir = os.path.join("..","..","reports", vectorizer_name, classifier_name)
        os.makedirs(results_dir, exist_ok=True)

        # Save the evaluation results in a text file
        with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    pipeline = Pipeline(raw_data_path= r"C:\Users\svenb\Desktop\MLApps\data\merged_reviews.csv", 
                        preprocess_data_path= r"C:\Users\svenb\Desktop\MLApps\data\preprocessed_data2.csv",
                        load_preprocessed=True,
                        preprocess_text_column="cleaned_text")
    pipeline.execute()