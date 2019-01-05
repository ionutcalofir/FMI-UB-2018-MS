import re
import time

from nltk.book import text2

def c2():
    print('Lungimea textului: ', len(text2.tokens))

def c3():
    print('Titlul: ', text2.name)

def c4():
    no_w = 0
    for w in text2.vocab():
        if w.isalpha() and not bool(re.search(r'[!.?]', w)):
            no_w += 1

    print('Lungimea vocabularului: ', no_w)

def c5(L=5):
    no_w = 0
    for w in text2.tokens:
        if len(w) == L:
            no_w += 1

    print('Numarul de cuvinte care au lungimea {0} este: {1}'.format(L, no_w))

def c6(N=5):
    words = [k for k, v in text2.vocab().items()]
    words.sort()

    for w in words:
        if w.startswith('L'):
            print(w)
            N -= 1
            if N == 0:
                break

def c7():
    min_w = len(text2.tokens[0])
    max_w = len(text2.tokens[0])

    for w in text2.tokens:
        min_w = min(min_w, len(w))
        max_w = max(max_w, len(w))

    print('Min: ', min_w, 'Max: ', max_w)

    minw = {}
    print('Cele mai mici cuvinte:')
    for w in text2.tokens:
        if len(w) == min_w:
            minw[w] = True
    print(minw.keys())

    maxw = {}
    print('Cele mai mari cuvinte:')
    for w in text2.tokens:
        if len(w) == max_w:
            maxw[w] = True
    print(maxw.keys())

def c8(N=5):
    words = [(w, text2.vocab()[w]) for w in text2.vocab()]
    words.sort(key=lambda k : k[1], reverse=True)

    print('Cele mai frecvente {0} cuvinte'.format(N))
    print(words[:N])

def c9():
    t = 0
    for w in text2.tokens:
        t += len(w)

    print('Lungimea medie a cuvintelor din text este {0}'.format(t / len(text2.tokens)))

def c10():
    for w in text2.vocab():
        if text2.vocab()[w] == 1:
            print(w)

def c11():
    text2.collocations()

if __name__ == '__main__':
    start_time = time.time()
    c2()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c3()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c4()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c5()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c6()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c7()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c8()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c9()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c10()
    print("--- %s seconds ---" % (time.time() - start_time))

    print('--------------------------------------------------------------------')

    start_time = time.time()
    c11()
    print("--- %s seconds ---" % (time.time() - start_time))
