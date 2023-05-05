import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.matutils import corpus2csc

class BoWVectorizer:
    def __init__(self, filename, text_column, method='count', no_below=4, no_above=0.8):
        self.filename = filename
        self.text_column = text_column
        self.method = method
        self.no_below = no_below
        self.no_above = no_above
        self.dictionary = None
        self.corpus_transformed = None

    def _read_csv(self):
        df = pd.read_csv(self.filename, encoding='utf-8')
        corpus = df[self.text_column].apply(lambda x: x.split()).tolist()
        return corpus

    def fit_transform(self):
        print(f"Vectorizing with BoW {self.method}")
        corpus = self._read_csv()
        
        self.dictionary = Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=self.no_below, no_above=self.no_above)
        
        if self.method == 'count':
            self.corpus_transformed = [self.dictionary.doc2bow(doc) for doc in corpus]
        elif self.method == 'tfidf':
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in corpus]
            tfidf_model = TfidfModel(bow_corpus)
            self.corpus_transformed = tfidf_model[bow_corpus]
        else:
            raise ValueError("Invalid method. Available methods are 'count' and 'tfidf'.")
        
        return corpus2csc(self.corpus_transformed).T

    def get_sparse_matrix(self):
        return corpus2csc(self.corpus_transformed)
