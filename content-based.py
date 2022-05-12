import numpy as np
import pandas as pd
from config import settings
import re
from stop_words import get_stop_words

text_df = pd.read_csv(settings['content_file'])
text_df_sample = text_df[:10000]

print(*get_stop_words(language="en"), sep='\n')

splitters = [';', ',', ' ', ';', ':']

words = set()
for x in text_df_sample["blurb"]:
    text = ''.join(i if (i.isalnum()) or (i == " ") else " " for i in x)
    text = text.split()
    for w in text:
        # w = w.replace(",", '').replace()
        words.add(w)

for x in get_stop_words(language="en"):
    words.discard(x)
words = sorted(words)

print(words)
