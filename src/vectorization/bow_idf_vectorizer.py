import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from gensim.matutils import corpus2csc

class BoWVectorizer:
    def __init__(self, corpus, method='count', no_below=4, no_above=0.8):
        self.method = method
        self.dictionary = Dictionary(corpus)
        self.dictionary.filter_extremes(no_below=no_below, no_above=no_above)
        
        if method == 'count':
            self.corpus_transformed = [self.dictionary.doc2bow(doc) for doc in corpus]
        elif method == 'tfidf':
            bow_corpus = [self.dictionary.doc2bow(doc) for doc in corpus]
            tfidf_model = TfidfModel(bow_corpus)
            self.corpus_transformed = tfidf_model[bow_corpus]
        else:
            raise ValueError("Invalid method. Available methods are 'count' and 'tfidf'.")
            
    def get_sparse_matrix(self):
        return corpus2csc(self.corpus_transformed)
