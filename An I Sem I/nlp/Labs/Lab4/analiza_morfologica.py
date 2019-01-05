import nltk
import time
import matplotlib.pyplot as plt
import numpy as np

from nltk.tag.stanford import StanfordPOSTagger 
from nltk.book import text2

tagger = StanfordPOSTagger('stanford-postagger-full-2018-10-16/models/english-bidirectional-distsim.tagger',
                           'stanford-postagger-full-2018-10-16/stanford-postagger.jar')

def c1(s='I saw a cat running after a mouse'):
    s_tokens = nltk.word_tokenize(s)
    tag = tagger.tag(s_tokens)

    print(tag)

def c2(ptag='NN'):
    tag = tagger.tag(text2.tokens[:100])
    for t in tag:
        if t[1] == ptag:
            print(t[0])

def c3(ptags=['NN', 'VB']):
    tag = tagger.tag(text2.tokens[:100])
    for t in tag:
        if t[1] in ptags:
            print(t[0])

def c4(N=5):
    d = {}
    tag = tagger.tag(text2.tokens[:100])
    for t in tag:
        if t[1] not in d:
            d[t[1]] = 1
        else:
            d[t[1]] += 1

    words = [(k, v) for k, v in d.items()]
    words.sort(key=lambda k : k[1], reverse=True)

    names = [t[0] for t in words]
    val = [t[1] for t in words]

    plt.bar(np.arange(len(val)), val)
    plt.xticks(np.arange(len(names)), names)

    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    # c1()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    # c2()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c3()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c4()
    print("--- %s seconds ---" % (time.time() - start_time))
