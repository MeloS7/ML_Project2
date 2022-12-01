from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

class Vectorization:

    def __init__(self, corpus, train_doc, test_doc):
        self.corpus = corpus
        self.train_doc = train_doc
        self.test_doc = test_doc

    def Counter_Vec(self):
        count_vec = CountVectorizer()
        count_vec_train = count_vec.fit(self.corpus)
        train_x_count_vec = count_vec_train.transform(self.train_doc)
        test_x_count_vec = count_vec_train.transform(self.test_doc)
        return train_x_count_vec, test_x_count_vec

    def Tf_idf(self):
        tfidf_vec = TfidfVectorizer()
        tfidf_vec_train = tfidf_vec.fit(self.corpus)
        train_x_tfidf = tfidf_vec_train.transform(self.train_doc)
        test_x_tfidf = tfidf_vec_train.transform(self.test_doc)
        return train_x_tfidf, test_x_tfidf
