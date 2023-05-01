import numpy as np
from gensim.models import Word2Vec

class IterableSentenceFromFile(object):
    def __init__(self, filename):
        self.__filename = filename
    
    def __iter__(self):
        for line in open(self.__filename):
            # assume there's one sentence per line, tokens separated by whitespace
            yield line.split()

class Word2VecVectorizer:
    def __init__(self, vector_size=200, window=5, seed=42, sg=1):
        self.vector_size = vector_size
        self.window = window
        self.seed = seed
        self.sg = sg

    def fit_transform(self, corpus, sentences_filename):
        sentences = IterableSentenceFromFile(sentences_filename)
        self.model_w2v = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, seed=self.seed, sg=self.sg)

        n_reviews = len(corpus)
        embedding_matrix = np.zeros((n_reviews, self.vector_size))
        for i, review in enumerate(corpus):
            embedding_matrix[i] = self.get_review_vector(review)

        return embedding_matrix

    def get_review_vector(self, review):
        vec_sum = np.zeros(self.vector_size,)
        n_tokens = 0
        for token in review:
            if token in self.model_w2v.wv:
                vec_sum += self.model_w2v.wv[token]
                n_tokens += 1

        if n_tokens > 0:
            return vec_sum / n_tokens
        else:
            return vec_sum
