import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from contractions import fix
import pandas as pd

class TextPreprocessor:
    def __init__(self,input_path, remove_stopwords=True, lemmatize=True):
        self.data_df = pd.read_csv(input_path)
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def preprocess_text(self, text):


        # Remove non-alphanumeric characters
        text = re.sub(r'\W', ' ', text)

        # Lowercase text
        text = text.lower()

        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text)

        # Remove HTML tags 
        text = BeautifulSoup(text, "lxml").get_text()
        
        # Remove URLs 
        text = re.sub(r'https://\S+|www\.\S+', '', text)
        
        # Expand contractions using the contractions library
        text = fix(text)

        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        if self.remove_stopwords:
            tokens = [tok for tok in tokens if tok not in self.stopwords]

        # Lemmatize tokens
        if self.lemmatize:
            tokens = [self.lemmatizer.lemmatize(tok) for tok in tokens]

        # Join tokens back into a single string
        cleaned_text = ' '.join(tokens)

        return cleaned_text

    def preprocess_corpus(self, output_filename, text_column="text", label_column="rating"):
        print("Preprocessing Text Corpus...")
        self.data_df.dropna(subset=[text_column], inplace=True)
        self.data_df.dropna(subset=[label_column], inplace=True)
        corpus = self.data_df[text_column]
        labels = self.data_df[label_column]

        preprocessed_corpus = [nltk.word_tokenize(self.preprocess_text(text)) for text in corpus]
        preprocessed_corpus = [' '.join(tokens) for tokens in preprocessed_corpus]

        output_df = pd.DataFrame({'cleaned_text': preprocessed_corpus, 'labels': labels})
        output_df.to_csv(output_filename, index=False, encoding='utf-8')

        print("Done")
        print(f"Length of preprocessed corpus in preprocesser:{len(preprocessed_corpus)}")
        print(f"Length of labels in preprocesser:{len(labels)}")
        return preprocessed_corpus, labels
    

    def preprocess_labels(self, label_column="rating"):
        print("Cleaning Labels...")

        numeric_pattern = re.compile(r'^\d+$')

        def to_integer(x):
            if numeric_pattern.match(x):
                return int(x)
            else:
                return None

        self.data_df[label_column] = self.data_df[label_column].apply(to_integer)
        self.data_df = self.data_df[self.data_df[label_column].notnull()]
        self.data_df.dropna(subset=[label_column], inplace=True)

        print("Done")

        return 

if __name__ == "__main__":
    
    in_path = os.path.join("..","..","data","merged_reviews.csv")
    out_path = os.path.join("..","..","data","preprocessed_data2.csv")
    preprocessor = TextPreprocessor(input_path=in_path)
    preprocessor.preprocess_labels()
    preprocessor.preprocess_corpus(out_path)