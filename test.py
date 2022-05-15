import numpy as np
import pandas as pd
from config import settings
from stop_words import get_stop_words as sw
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

text_df = pd.read_csv(settings['content_file'])
n = 1000
text_df_sample = text_df[:n]

texts = list(text_df_sample['blurb'])
states = list(text_df_sample['state'])

texts_tf_idf = {}
texts_lemma = []
for ind in range(text_df_sample.shape[0]):
    words = {}
    sent = texts[ind]
    text = ''.join(s.lower() if s.isalnum() else " " for s in sent)
    text = nlp(text)
    for w in text:
        if (not w.lemma_.isdecimal()) and (len(w) > 1):
            w_lemma = w.lemma_
            if w_lemma in words:
                words[w_lemma] += 1
            else:
                words.update({w_lemma: 1})
    for x in sw(language='en'):
        words.pop(x, None)
    texts_lemma.append(words)

i = 1
for words in texts_lemma:
    words_unique = len(words)
    text_tf_idf = {}
    texts_with_word = 0
    for word in words:
        tf = words[word] / words_unique
        for text_check in texts_lemma:
            if word in text_check:
                texts_with_word += 1
        idf = np.log10(n/texts_with_word)
        tf_idf = tf * idf
        text_tf_idf.update({word: tf_idf})
    texts_tf_idf.update({i: text_tf_idf})
    i += 1


print(texts_tf_idf)
