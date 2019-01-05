import time
import re
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

class Dataset():
    def __init__(self,
                 data_path='data/Calofir A. Petrișor-Ionuț.csv'):
        self._data_path = data_path
        self._stopwords = stopwords.words('english')
        self._wn_lemma = WordNetLemmatizer()

    def tags_to_wn(self, tag):
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return wordnet.NOUN
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return wordnet.VERB

        return -1

    def build_tagger_and_lemma_processed_data(self):
        data_csv = pd.read_csv(self._data_path)

        with open('data/tagger_and_lemma_processed_data.txt', 'w') as f:
            for i, col in enumerate(data_csv.keys()[1:]):
                print(i, col)
                for text in data_csv[col]:
                    text = word_tokenize(text)
                    start = time.time()
                    text = pos_tag(text)
                    text = [t for t in text if self.tags_to_wn(t[1]) != -1]
                    text = [self._wn_lemma.lemmatize(t[0], pos=self.tags_to_wn(t[1])) for t in text]
                    end = time.time()
                    print('Time elapsed for StanfordTagger: {0}'.format(end - start))

                    f.write(json.dumps(text))
                    f.write('\n')

    def build_processed_data(self):
        with open('data/tagger_and_lemma_processed_data.txt', 'r') as f:
            with open('data/processed_data.txt', 'w') as fl:
                for i, line in enumerate(f):
                    print('Process text number: {0}'.format(i))
                    text = json.loads(line)

                    text = [word.lower() for word in text
                            if word.lower() not in self._stopwords # remove stopwords
                                and len(word) > 2 # remove words that contain less than 2 characters
                                and not bool(re.search(r'\d', word))] # check if the word contains any numbers

                    fl.write(json.dumps(text))
                    fl.write('\n')

    def get_data_classification(self):
        X = []
        with open('data/processed_data.txt', 'r') as f:
            for i, line in enumerate(f):
                # print('Read text {0}'.format(i))
                text = json.loads(line)
                X.append(' '.join(text))

        data_csv = pd.read_csv(self._data_path)
        y = []
        self._author_to_class = {}
        with open('data/tagger_and_lemma_processed_data.txt', 'w') as f:
            for i, col in enumerate(data_csv.keys()[1:]):
                # print('Class for text {0}'.format(i))
                self._author_to_class[col] = i
                aux = data_csv[col].shape[0] * [i]
                y.extend(aux)

        X = np.array(X)
        y = np.array(y)

        X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.15, random_state=1)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, stratify=y_train, test_size=(y_val.shape[0]) / y_train.shape[0], random_state=1)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def get_data_clustering(self):
        X = []
        with open('data/processed_data.txt', 'r') as f:
            for i, line in enumerate(f):
                # print('Read text {0}'.format(i))
                text = json.loads(line)
                X.append(' '.join(text))

        data_csv = pd.read_csv(self._data_path)
        y = []
        self._author_to_class = {}
        self._class_to_author = {}
        with open('data/tagger_and_lemma_processed_data.txt', 'w') as f:
            for i, col in enumerate(data_csv.keys()[1:]):
                # print('Class for text {0}'.format(i))
                self._author_to_class[col] = i
                self._class_to_author[i] = col
                aux = data_csv[col].shape[0] * [i]
                y.extend(aux)

        X = np.array(X)
        y = np.array(y)

        return X, y
