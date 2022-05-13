import numpy as np
import pandas as pd
from config import settings
from textblob import Word
from stop_words import get_stop_words as sw

text_df = pd.read_csv(settings['content_file'])
text_df_sample = text_df[:10000]

texts = list(text_df_sample['blurb'])
states = list(text_df_sample['state'])

words = {}
for ind in range(text_df_sample.shape[0]):
    sent = texts[ind]
    text = ''.join(s if s.isalnum() else " " for s in sent).split()
    for w in text:
        if not w.isdecimal() and len(w) > 1:
            w = Word(w.lower()).lemmatize()
            state = 1 if states[ind] == 'successful' else 0
            if w in words:
                words[w][0] += 1
                words[w][1] += state
            else:
                words.update({w: [1, state]})

for x in sw(language='en'):
    words.pop(x, None)

print(sorted(words.items(), key=lambda el: el[1][0], reverse=True))
