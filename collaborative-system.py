import numpy as np
import pandas as pd

ratings_df = pd.read_csv('ml-latest-small/ml-latest-small/ratings.csv')
print('Unique users count: {}'.format(len(ratings_df['userId'].unique())))
print('Unique movies count: {}'.format(len(ratings_df['movieId'].unique())))
print('DataFrame shape: {}'.format(ratings_df.shape))

ratings_df.head()