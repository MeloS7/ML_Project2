import string
from string import digits
import nltk
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('averaged_perceptron_tagger')
# nltk.download('wordnet')
# nltk.download('omw-1.4')


class Pre_processing:

    def __init__(self, data):
        self.data = data
        self.punct = str.maketrans({key:"" for key in string.punctuation})
        self.no_digit = str.maketrans('', '', digits)
        data_cleaned = []
        word_corpus = []

    def get_wordnet_pos(self, tag):
        if tag.startswith('J'):
            return wordnet.ADJ
        elif tag.startswith('V'):
            return wordnet.VERB
        elif tag.startswith('N'):
            return wordnet.NOUN
        elif tag.startswith('R'):
            return wordnet.ADV
        else:
            return None

    def data_clean(self, stemming=False, for_test=False):
        data_cleaned = []
        if not stemming:
            for line in self.data:
                # Remove HTML element such as <h>
                line_be = BeautifulSoup(line, 'html.parser').get_text()

                # Remove punctuations
                # line_clean = line_be.translate(self.punct)
                
                # Remove digitals
                # line_clean_no_number = line_clean.translate(self.no_digit)
                
                # Make all words into lower case and split them
                # line_split = line_clean_no_number.lower()

                # Tokenize
                words_tokenized = word_tokenize(line_be)
                
                # Remove Stopwords in English
                # stops = set(stopwords.words("english"))
                # words = [w for w in words_tokenized if not w in stops]

                words = [w for w in words_tokenized]
            
                if for_test:
                    # Remove indices in test dataset
                    data_cleaned.append(words[2:])
                else:
                    data_cleaned.append(words)
        else:
            for line in self.data:
                # Remove HTML element such as <h>
                line_be = BeautifulSoup(line, 'html.parser').get_text()

                # Remove punctuations
                line_clean = line_be.translate(self.punct)
                
                # Remove digitals
                line_clean_no_number = line_clean.translate(self.no_digit)
                
                # Make all words into lower case and split them
                line_split = line_clean_no_number.lower()

                # Tokenize
                words_tokenized = word_tokenize(line_split)

                # Stemming
                tagged_sent = pos_tag(words_tokenized)
                wnl = WordNetLemmatizer()
                lemmas_sent = []
                for tag in tagged_sent:
                    wordnet_pos = self.get_wordnet_pos(tag[1]) or wordnet.NOUN
                    lemmas_sent.append(wnl.lemmatize(tag[0], pos=wordnet_pos))
                
                # Remove Stopwords in English
                stops = set(stopwords.words("english"))
                words = [w for w in lemmas_sent if not w in stops]

                if for_test:
                    # Remove indices in test dataset
                    data_cleaned.append(words[2:])
                else:
                    data_cleaned.append(words)
            
            
        self.data_cleaned = data_cleaned
        return self.data_cleaned

    def create_corpus(self):
        corpus = []
        for sen in self.data_cleaned:
            for w in sen:
                if w not in corpus:
                    corpus.append(w)
        self.word_corpus = corpus
        return self.word_corpus

    def create_document(self):
        self.document = [' '.join(s) for s in self.data_cleaned]
        return self.document

    def streamline(self, stemming=False, for_test=False):
        self.data_clean(stemming, for_test)
        self.create_corpus()
        self.create_document()
        return self.word_corpus, self.document, self.data_cleaned

    def add_label(self, stemming=False, for_test=False):
        '''
        This function is for fasttext training.
        Add labels in the documents to match fasttext training set format.
        '''
        self.data_clean(stemming, for_test)
        self.create_document()
        for i in range(len(self.document)):
            if i < len(self.document)//2:
                self.document[i] = '__label__neg , ' + self.document[i]
            else:
                self.document[i] = '__label__pos , ' + self.document[i]
        return self.document