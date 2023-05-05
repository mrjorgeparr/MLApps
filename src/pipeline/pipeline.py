import os
import sys
import sys
import json
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from preprocessing.text_preprocesser import TextPreprocessor
from vectorization.bow_idf_vectorizer import BoWVectorizer
from vectorization.word2vec_vectorizer import Word2VecVectorizer

from machine_learning.logistic_regression_classifier import LogisticRegressionClassifier
from machine_learning.knn_classifier import KNNClassifier

from sklearn.model_selection import train_test_split

class Pipeline:
    def __init__(self, raw_data_path, processed_data_path):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.labels = None

        # Initialize components
        self.preprocessor = TextPreprocessor()
        self.vectorizers = {
            "bow": BoWVectorizer(),
            "word2vec": Word2VecVectorizer()
        }
        self.classifiers = {"Logistic Regression": LogisticRegressionClassifier(random_state=42,),
                            "KNN": KNNClassifier(random_state=42)}  # More classifiers can be added here

    def add_preprocessor(self, preprocessor):
        self.preprocessor = preprocessor

    def add_vectorizer(self, vectorizer_name, vectorizer):
        self.vectorizers[vectorizer_name] = vectorizer

    def add_classifier(self, classifier_name, classifier):
        self.classifiers[classifier_name] = classifier

    def preprocess(self):

        output_filename = os.path.join(self.processed_data_path, 'preprocessed_data.csv')
        self.preprocessed_data, self.labels = self.preprocessor.preprocess_corpus(self.raw_data_path, output_filename)


    def execute(self):
        self.preprocess()
        print(f"Length of preprocessed_data: {len(self.preprocessed_data)}")
        print(f"Length of labels: {len(self.labels)}")

        # Vectorize the data
        for vectorizer_name, vectorizer in self.vectorizers.items():
            if vectorizer_name == "word2vec":
                sentences_filename = os.path.join(self.processed_data_path, 'preprocessed_data.csv')
                vectorized_data = vectorizer.fit_transform(self.preprocessed_data, sentences_filename)

            else:
                vectorized_data = vectorizer.fit_transform(self.preprocessed_data)

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

                # Evaluate the classifier
                eval_results = classifier.evaluate(X_test, y_test, vectorizer_name)

                # Save the results
                self.save_results(vectorizer_name, classifier_name, eval_results)

    def save_results(self, vectorizer_name, classifier_name, eval_results):
        # Define a directory for saving the results
        results_dir = os.path.join("..","..","reports", vectorizer_name, classifier_name)
        os.makedirs(results_dir, exist_ok=True)

        # Save the evaluation results in a text file
        with open(os.path.join(results_dir, "evaluation_results.txt"), "w") as f:
            for key, value in eval_results.items():
                f.write(f"{key}: {value}\n")


if __name__ == "__main__":
    pipeline = Pipeline(r"C:\Users\svenb\Desktop\MLApps\data\merged_reviews.csv", r"C:\Users\svenb\Desktop\MLApps\src\data_collection")
    pipeline.execute()