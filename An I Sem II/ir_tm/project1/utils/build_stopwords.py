from nltk.corpus import stopwords

# unicode chars
map_chars = {351: [351, 537],
             355: [355, 539],
             537: [537, 351],
             539: [539, 355],
             238: [238, 226]}

def make_all_words(pos, new_s, s, f):
    if pos == len(s):
        f.write(new_s)
        f.write('\n')
        return

    if ord(s[pos]) in map_chars:
        for c in map_chars[ord(s[pos])]:
            make_all_words(pos + 1, new_s + chr(c), s, f)
    else:
        make_all_words(pos + 1, new_s + s[pos], s, f)

stopwords_ro = stopwords.words('romanian')

with open('stopwords.txt', 'w', encoding='utf-8') as f:
    for word in stopwords_ro:
        make_all_words(0, '', word, f)
