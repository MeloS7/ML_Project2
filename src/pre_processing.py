import string
from string import digits
import nltk
nltk.download('stopwords')
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class Pre_processing:

    def __init__(self, data):
        self.data = data
        self.punct = str.maketrans({key:"" for key in string.punctuation})
        self.no_digit = str.maketrans('', '', digits)
        data_cleaned = []
        word_corpus = []

    def data_clean(self):
        data_cleaned = []
        for line in self.data:
            # Remove HTML element such as <h>
            line_be = BeautifulSoup(line, 'html.parser').get_text()

            # Remove punctuations
            line_clean = line_be.translate(self.punct)
            
            # Remove digitals
            line_clean_no_number = line_clean.translate(self.no_digit)
            
            # Make all words into lower case and split them
            line_split = line_clean_no_number.lower().split()
            
            # Remove Stopwords in English
            stops = set(stopwords.words("english"))
            words = [w for w in line_split if not w in stops]
            
            data_cleaned.append(words)
        self.data_cleaned = data_cleaned
        return self.data_cleaned

    def create_corpus(self):
        corpus = []
        for sen in self.data_cleaned:
            for w in sen:
                corpus.append(w)
        self.word_corpus = corpus
        return self.word_corpus

    def create_document(self):
        self.document = [' '.join(s) for s in self.data_cleaned]
        return self.document

    def streamline(self):
        self.data_clean()
        self.create_corpus()
        self.create_document()
        return self.word_corpus, self.document