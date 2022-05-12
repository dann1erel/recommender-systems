import numpy as np
import pandas as pd
from config import settings
from stop_words import get_stop_words as sw

text_df = pd.read_csv(settings['content_file'])
text_df_sample = text_df[:10000]

words = {}
for x in text_df_sample["blurb"]:
    text = ''.join(i if (i.isalnum()) or (i == " ") else " " for i in x).split()
    for w in text:
        if len(w) > 1:
            w = w.lower()
            if w in words:
                words[w] += 1
            else:
                words.update({w: 1})

for x in sw(language='en'):
    words.pop(x, None)

print(sorted(words.items(), key=lambda el: el[1], reverse=True))
