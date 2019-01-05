import nltk
import time

from nltk.tag.stanford import StanfordPOSTagger 
from nltk.corpus import stopwords

tagger = StanfordPOSTagger('stanford-postagger-full-2018-10-16/models/english-bidirectional-distsim.tagger',
                           'stanford-postagger-full-2018-10-16/stanford-postagger.jar')

if __name__ == '__main__':
    start_time = time.time()
    with open('take_morfo.txt', 'r') as f:
        text = f.read()

    text_token = nltk.word_tokenize(text)

    with open('out.txt', 'w') as f:
        f.write(str(text_token))
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    with open('out.txt', 'a') as f:
        f.write('\n\n')

        sents = nltk.sent_tokenize(text)
        f.write('Numarul de propozitii este {0}'.format(len(sents)))
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    with open('out.txt', 'a') as f:
        f.write('\n\n')
        sents = nltk.sent_tokenize(text)

        for sent in sents:
            f.write(sent + '\n')

            words = nltk.word_tokenize(sent)
            f.write('Numarul de cuvinte {0}\n'.format(len(words)))

            sw = stopwords.words('english')
            no_sw = 0
            for w in words:
                if w in sw:
                    no_sw += 1
            f.write('Numarul de stopwords {0}\n'.format(no_sw))

            for w in words:
                if w not in sw:
                    f.write(w + ' ')

            f.write('\n')
            tag = tagger.tag(words)
            f.write(str(tag) + '\n')

            wt = 0
            for t in tag:
                if t[1] == 'NN' or t[1] == 'VB':
                    wt += 1
            f.write(str(wt / len(words)) + '\n')

            f.write('\n')
    print("--- %s seconds ---" % (time.time() - start_time))
