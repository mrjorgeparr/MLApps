import numpy as np
from gensim.models import Word2Vec
import csv

class IterableSentenceFromFile(object):
    def __init__(self, filename, text_column='cleaned_text', n=1):
        self.filename = filename
        self.text_column = text_column
        self.n = n

    def __iter__(self):
        with open(self.filename, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                tokens = row[self.text_column].split()
                yield self.get_ngrams(tokens)

    def get_ngrams(self, tokens):
        ngrams = []
        for n in range(1, self.n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = ' '.join(tokens[i:i+n])
                ngrams.append(ngram)
        return ngrams

class Word2VecVectorizer:
    def __init__(self, filename, text_column, vector_size=200, window=5, seed=42, sg=1, n=1):
        self.filename = filename
        self.text_column = text_column
        self.vector_size = vector_size
        self.window = window
        self.seed = seed
        self.sg = sg
        self.n = n

    def fit_transform(self):
        print(f"Vectorizing with word2vec {self.n}-gram and window size: {self.window}")
        sentences = IterableSentenceFromFile(self.filename, text_column=self.text_column, n=self.n)
        self.model_w2v = Word2Vec(sentences, vector_size=self.vector_size, window=self.window, seed=self.seed, sg=self.sg)

        embedding_matrix = []
        for sentence in sentences:
            review_vector = self.get_review_vector(sentence)
            embedding_matrix.append(review_vector)

        return np.array(embedding_matrix)


    def get_review_vector(self, review):
        vec_sum = np.zeros(self.vector_size,)
        n_tokens = 0
        iterable_sentence = IterableSentenceFromFile(self.filename, text_column=self.text_column, n=self.n)
        ngrams = iterable_sentence.get_ngrams(review)
        for ngram in ngrams:
            if ngram in self.model_w2v.wv:
                vec_sum += self.model_w2v.wv[ngram]
                n_tokens += 1

        if n_tokens > 0:
            return vec_sum / n_tokens
        else:
            return vec_sum
