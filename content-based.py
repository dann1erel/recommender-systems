import numpy as np
import pandas as pd
from config import settings
from stop_words import get_stop_words as sw
import spacy
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

text_df = pd.read_csv(settings['content_file'])
n = 3000
text_df_sample = text_df[:n]

texts = list(text_df_sample['blurb'])
states = list(text_df_sample['state'])

all_words = {}
texts_lemma = []
user_text = input("Type in your project description: ")


def transform_text(sent):
    words_in_text = {}
    text = ''.join(s.lower() if s.isalnum() else " " for s in sent)
    text = nlp(text)
    for w in text:
        if (not w.lemma_.isdecimal()) and (len(w) > 1):
            w_lemma = w.lemma_
            if ' ' not in w_lemma:
                if w_lemma in words_in_text:
                    words_in_text[w_lemma] += 1
                else:
                    words_in_text.update({w_lemma: 1})
                if w_lemma not in all_words:
                    all_words.update({w_lemma: 0})
    for x in sw(language='en'):
        words_in_text.pop(x, None)
        all_words.pop(x, None)
    texts_lemma.append(words_in_text)


def cos_similarity(vec_a, vec_b):
    # dot - scalar product, norm - vector length
    res = (np.dot(vec_a, vec_b))/(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return res


for ind in range(text_df_sample.shape[0]):
    transform_text(texts[ind])
transform_text(user_text)

texts_tf_idf = {}
for i in range(n+1):
    all_words_c = all_words.copy()
    texts_tf_idf.update({i: all_words_c})
i = 0
for words in texts_lemma:
    words_unique = len(words)
    text_tf_idf = {}
    for word in words:
        texts_with_word = 0
        tf = words[word] / words_unique
        for text_check in texts_lemma:
            if word in text_check:
                texts_with_word += 1
        idf = np.log10(n/texts_with_word)
        tf_idf = tf * idf
        text_tf_idf.update({word: tf_idf})
    for word in text_tf_idf:
        texts_tf_idf[i][word] = text_tf_idf[word]
    i += 1

# print(texts_tf_idf)

vectors = []
for i in texts_tf_idf:
    sentence = texts_tf_idf[i]
    vector = np.array([sentence[i] for i in sentence])
    vectors.append(vector)
user_vector = vectors[-1]
vectors.pop(-1)

cosines = []
for vec in vectors:
    cosines.append(cos_similarity(vec, user_vector))
cosines_enum = tuple(sorted(enumerate(cosines), key=lambda el: el[1], reverse=True))

sim_border = 0.15

i = 0
result = 0
while cosines_enum[i][1] >= sim_border:
    if text_df_sample['state'][cosines_enum[i][0]] == 'successful':
        result += 1 * cosines_enum[i][1]
    else:
        result -= 1 * cosines_enum[i][1]
    i = i + 1

if result < 0:
    print("This project most likely will fail")
elif result > 0.2:
    print("This project most likely will be successful")
else:
    print("Predictions cannot be made")
