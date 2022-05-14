import numpy as np
import pandas as pd
from config import settings
from stop_words import get_stop_words as sw
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

text_df = pd.read_csv(settings['content_file'])
text_df_sample = text_df[:100]

texts = list(text_df_sample['blurb'])
states = list(text_df_sample['state'])

texts_tf_idf = {}
texts_lematized = []
for ind in range(text_df_sample.shape[0]):
    words = {}
    sent = texts[ind]
    text = ''.join(s.lower() if s.isalnum() else " " for s in sent)
    text = nlp(text)
    for w in text:
        if (not w.lemma_.isdecimal()) and (len(w) > 1):    #🤯🥴🤢
            w_lemmatized = w.lemma_
            if w_lemmatized in words:
                words[w_lemmatized] += 1
            else:
                words.update({w_lemmatized: 1})
    for x in sw(language='en'):
        words.pop(x, None)
    texts_lematized.append(words)
for words in texts_lematized:
    words_unique = len(words)
    for word in words:
        for text_check in texts_lematized:
            if word in
            tf = words[word] / words_unique




print(sorted(words.items(), key=lambda el: el[1][0], reverse=True))
