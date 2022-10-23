import torch
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
from sklearn.model_selection import train_test_split
import numpy as np

from gensim.models import KeyedVectors

class Dataset():
  def __init__(self, train_config_path):
    self.train_config_path = train_config_path

    print('Load word2vec')
    self.word2vec = KeyedVectors.load('./data/word2vec_fast/word2vec', mmap='r')

    # Run the below code only once to enable fast loading for word2vec model
    # self.word2vec = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    # self.word2vec.init_sims(replace=True)
    # self.word2vec.save('./data/word2vec_fast/word2vec')
    print('Done')

    self.lemma = WordNetLemmatizer() 

    self._word_to_idx = {}
    self._weights = None
    self._build_vocab()

  def _preprocess_data_text(self, sentence, words):
    # text = sentence + ' _blank_ ' + words + ' _blank_ ' + sentence
    text = '_start_ ' + words + ' _start_sent_ ' + sentence + ' _end_'

    return text

  def _preprocess_sentence(self, sentence):
    words = word_tokenize(sentence)
    # words = [word.lower() for word in words if len(word) > 3]
    words = [self.lemma.lemmatize(word.lower()) for word in words if len(word) > 3]

    return words

  def _build_vocab(self):
    self._word_to_idx['_blank_'] = 0
    self._word_to_idx['_start_'] = 1
    self._word_to_idx['_start_sent_'] = 2
    self._word_to_idx['_end_'] = 3
    self._word_to_idx['_unk_'] = 4

    with open(self.train_config_path, 'r') as f:
      for line in f:
        data = line.strip().split('\t')
        data_text = self._preprocess_data_text(data[1], data[4])
        words = self._preprocess_sentence(data_text)

        for word in words:
          if word not in self._word_to_idx and word in self.word2vec:
            self._word_to_idx[word] = len(self._word_to_idx)

    self._weights = np.random.rand(len(self._word_to_idx), 300)
    for k, v in self._word_to_idx.items():
      if k in self.word2vec:
        self._weights[v][:] = self.word2vec[k][:]
    self._weights[0][:] = np.zeros((300,))

  def sentenceToIdxs(self, sentence):
    words = self._preprocess_sentence(sentence)

    idxs = []
    for word in words:
      if word not in self._word_to_idx:
        idxs.append(4)
        continue
      idxs.append(self._word_to_idx[word])

    return idxs

  def get_data(self, train_config_path, test_config_path):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    with open(train_config_path, 'r') as f:
      for line in f:
        data = line.strip().split('\t')
        data_text = self._preprocess_data_text(data[1], data[4])
        X_train.append((data_text,
                        data[1],
                        data[4],
                        int(data[5]),
                        int(data[6]),
                        int(data[7]),
                        int(data[8])))
        y_train.append(float(data[9]))

    with open(test_config_path, 'r') as f:
      for line in f:
        data = line.strip().split('\t')
        data_text = self._preprocess_data_text(data[1], data[4])
        X_test.append((data_text,
                       data[1],
                       data[4],
                       int(data[5]),
                       int(data[6]),
                       int(data[7]),
                       int(data[8])))
        y_test.append(float(data[9]))

    return X_train, X_test, y_train, y_test

  def get_data_submission(self, submission_test_config_path):
    X = []

    with open(submission_test_config_path, 'r') as f:
      for line in f:
        data = line.strip().split('\t')
        data_text = self._preprocess_data_text(data[1], data[4])
        X.append((data_text,
                  data[1],
                  data[4],
                  int(data[5]),
                  data[0]))

    return X
