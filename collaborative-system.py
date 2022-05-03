import numpy as np
import pandas as pd
from config import settings

ratings_df = pd.read_csv(settings['collab_file'])


print('Unique users count: {}'.format(len(ratings_df['userId'].unique())))
print('Unique movies count: {}'.format(len(ratings_df['movieId'].unique())))
print('DataFrame shape: {}'.format(ratings_df.shape))
print("# {}".format(ratings_df.head()))

lst = np.array([[0.0]*ratings_df['movieId'].max()]*len(ratings_df['userId'].unique()))

for x in range(ratings_df.shape[0]):
    lst[list(ratings_df['userId'])[x]-1][list(ratings_df['movieId'])[x]-1] = list(ratings_df['rating'])[x]

print(*lst, sep='\n')

ratings_df.head()
