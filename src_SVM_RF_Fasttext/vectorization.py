from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sentence_transformers import SentenceTransformer

class Vectorization:

    def __init__(self, corpus, train_doc, name):
        self.corpus = corpus
        self.train_doc = train_doc
        self.name = name

    def Counter_Vec(self):
        count_vec = CountVectorizer()
        count_vec_train = count_vec.fit(self.corpus)
        train_x_count_vec = count_vec_train.transform(self.train_doc)
        return train_x_count_vec, count_vec_train

    def Tf_idf(self):
        tfidf_vec = TfidfVectorizer()
        tfidf_vec_train = tfidf_vec.fit(self.corpus)
        train_x_tfidf = tfidf_vec_train.transform(self.train_doc)
        return train_x_tfidf, tfidf_vec_train

    def N_gram(self, n):
        count_vec = CountVectorizer(ngram_range=(1, n))
        count_vec_train = count_vec.fit(self.corpus)
        train_x_count_vec = count_vec_train.transform(self.train_doc)
        return train_x_count_vec, count_vec_train

    def pre_trained_word_embedding(self):
        model_embedding = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        train_embeddings = model_embedding.encode(self.train_doc)
        return train_embeddings, model_embedding

    def select_by_name(self):
        if self.name == "CV":
            return self.Counter_Vec()
        elif self.name == "TF":
            return self.Tf_idf()
        elif self.name == "NGRAM":
            return self.N_gram(5)
        else:
            return self.pre_trained_word_embedding()
