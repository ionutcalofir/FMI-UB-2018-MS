import re
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.tag.stanford import StanfordPOSTagger 
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

class TextProcessing():
    def __init__(self):
        self._stopwords = stopwords.words('english')
        self._tokenizer = TreebankWordTokenizer()
        self._dw = {}
        self._cw = {}
        self._stanford_tagger = StanfordPOSTagger(
            'stanford-postagger-full-2018-10-16/models/english-bidirectional-distsim.tagger',
            'stanford-postagger-full-2018-10-16/stanford-postagger.jar')
        self._wn_lemma = WordNetLemmatizer()

    def stanford_to_wn(self, tag):
        if tag in ['NN', 'NNS', 'NNP', 'NNPS']:
            return wordnet.NOUN
        elif tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
            return wordnet.VERB

        return -1

    def analyze(self, text):
        text = self.process(text)
        text_id = self.word2id(text)

        return text_id

    def process(self, text):
        text = [self._process_document(doc) for doc in text]

        return text

    def _process_document(self, doc):
        text = self._tokenizer.tokenize(doc)
        text_tag = self._stanford_tagger.tag(text)
        text_nv = [t for t in text_tag if self.stanford_to_wn(t[1]) != -1]
        text = [self._wn_lemma.lemmatize(t[0], pos=self.stanford_to_wn(t[1])) for t in text_nv]
        self._lemma2word = {text[i] : text_nv[i][0] for i in range(len(text))}
        text = [word.lower() for word in text
                    if word.lower() not in self._stopwords # remove stopwords
                        and len(word) > 2 # remove words that contain less than 2 characters
                        and word != 'n\'t' # remove n't word from tokenization
                        and not bool(re.search(r'\d', word))] # check if the word contains any numbers

        return text

    def word2id(self, text):
        text_id = []
        if len(self._dw) == 0: # train
            for doc in text:
                ids = []
                for word in doc:
                    if not word in self._dw:
                        self._dw[word] = len(self._dw)
                        self._cw[self._dw[word]] = 1
                    else:
                        self._cw[self._dw[word]] += 1
                    ids.append(self._dw[word])
                text_id.append(ids)
        else:
            for doc in text:
                ids = []
                for word in doc:
                    if word in self._dw:
                        ids.append(self._dw[word])
                text_id.append(ids)

        return text_id

    def vocab_size(self):
        return len(self._dw)
