import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
from contractions import fix

class TextPreprocessor:
    def __init__(self, remove_stopwords=True, lemmatize=True):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def preprocess(self, text):


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

    def preprocess_all(self, texts):
        return [self.preprocess(text) for text in texts]
