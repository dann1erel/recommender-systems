import numpy as np
import pandas as pd
from config import settings
from stop_words import get_stop_words as sw
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

text_df = pd.read_csv(settings['content_file'])
text_df_sample = text_df[:10000]

texts = list(text_df_sample['blurb'])
states = list(text_df_sample['state'])

words = {}
for ind in range(text_df_sample.shape[0]):
    sent = texts[ind]
    text = ''.join(s.lower() if s.isalnum() else " " for s in sent)
    text = nlp(text)
    for w in text:
        if (not w.lemma_.isdecimal()) and (len(w) > 1):
            w_lemmatized = w.lemma_
            state = 1 if states[ind] == 'successful' else 0
            if ' ' not in w_lemmatized:
                if w_lemmatized in words:
                    words[w_lemmatized][0] += 1
                    words[w_lemmatized][1] += state
                else:
                    words.update({w_lemmatized: [1, state]})

for x in sw(language='en'):
    words.pop(x, None)

print(sorted(words.items(), key=lambda el: el[1][0], reverse=True))
