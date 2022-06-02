import spacy
import numpy as np
import pandas as pd
from config import settings
from stop_words import get_stop_words as sw

# Getting data for English text lemmatize
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])

# Reading csv file with pandas
text_df = pd.read_csv(settings['content_file'])

# Setting the size of sample
num_of_lines = 3000
text_df_sample = text_df[:num_of_lines]

# Saving columns in list for better perfomance
texts = list(text_df_sample['blurb'])
states = list(text_df_sample['state'])


# Setting new data storages

# Storage for all words
all_words = {}

# Storage for lemmatized sentences
texts_lemma = []

# Getting project description from user
user_text = input("Type in your project description: ")


# Lemmatizing sentence, filling storages
def transform_text(sent):

    # Words storage with amount
    words_in_text = {}
    # Cleaning text from unwanted words, switching to lowercase
    text = ''.join(s.lower() if s.isalnum() else " " for s in sent)
    # Lemmatizing text
    text = nlp(text)

    # Words processing
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

    # Cleaning text from meaningless(when out of context) words
    for x in sw(language='en'):
        words_in_text.pop(x, None)
        all_words.pop(x, None)

    texts_lemma.append(words_in_text)


# Cos similarity computation
def cos_similarity(vec_a, vec_b):
    # dot - scalar product, norm - vector length
    res = (np.dot(vec_a, vec_b))/(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return res


# Text processing

for ind in range(text_df_sample.shape[0]):
    transform_text(texts[ind])

transform_text(user_text)

# Storage for similarity coefficient
texts_tf_idf = {}
for i in range(num_of_lines + 1):
    all_words_c = all_words.copy()
    texts_tf_idf.update({i: all_words_c})

# Calculating tf_idf
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

        idf = np.log10(num_of_lines / texts_with_word)
        tf_idf = tf * idf
        text_tf_idf.update({word: tf_idf})

    for word in text_tf_idf:
        texts_tf_idf[i][word] = text_tf_idf[word]
    i += 1


# Creating vectors for cos similarity compare

vectors = []
for i in texts_tf_idf:
    sentence = texts_tf_idf[i]
    vector = np.array([sentence[i] for i in sentence])
    vectors.append(vector)

user_vector = vectors[-1]
vectors.pop(-1)

# Cosines storage
cosines = []
for vec in vectors:
    cosines.append(cos_similarity(vec, user_vector))

# Storage of sorted cosines
cosines_enum = tuple(sorted(enumerate(cosines), key=lambda el: el[1], reverse=True))

sim_border = 0.15

i = 0
result = 0
while cosines_enum[i][1] >= sim_border:
    if text_df_sample['state'][cosines_enum[i][0]] == 'successful':
        result += cosines_enum[i][1]
    else:
        result -= cosines_enum[i][1]

    i = i + 1

# Cos similarity compare, out results
if result < 0:
    print("This project most likely will fail")
elif result > 0.2:
    print("This project most likely will be successful")
else:
    print("Predictions cannot be made")
