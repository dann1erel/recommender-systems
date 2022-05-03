import numpy as np
import pandas as pd
from config import settings

ratings_df = pd.read_csv(settings['collab_file'])
print('Unique users count: {}'.format(len(ratings_df['userId'].unique())))
print('Unique movies count: {}'.format(len(ratings_df['movieId'].unique())))
print('DataFrame shape: {}'.format(ratings_df.shape))
print("# {}".format(ratings_df.head(n=100)))

ratings_df.head()