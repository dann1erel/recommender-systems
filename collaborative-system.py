import numpy as np
import pandas as pd
from config import settings

ratings_df = pd.read_csv(settings['collab_file'])

n = 100000
ratings_df_sample = ratings_df[:n]

n_users = len(ratings_df_sample['userId'].unique())
n_movies = len(ratings_df_sample['movieId'].unique())

print('Unique users count: {}'.format(len(ratings_df_sample['userId'].unique())))
print('Unique movies count: {}'.format(len(ratings_df_sample['movieId'].unique())))
print('DataFrame shape: {}'.format(ratings_df_sample.shape))
print("# {}".format(ratings_df.head()))

lst = np.array([[0.0]*ratings_df['movieId'].max()]*len(ratings_df['userId'].unique()))

for x in range(ratings_df.shape[0]):
    lst[tuple(ratings_df['userId'])[x]-1][tuple(ratings_df['movieId'])[x]-1] = tuple(ratings_df['rating'])[x]

print("\nRatings:")
print(*lst, sep='\n')

ratings_df.head()


def cos_similarity(vecA, vecB):
    result = (np.dot(vecA, vecB))/(np.linalg.norm(vecA) * np.linalg.norm(vecB))  # dot - scalar product, norm - vector length???
    return result


lst_sim = np.array([[0.0]*len(ratings_df['userId'].unique())]*len(ratings_df['userId'].unique()))

for i in range(len(ratings_df['userId'].unique())):
    for j in range(len(ratings_df['userId'].unique())):
        if i != j:
            lst_sim[i][j] = cos_similarity(np.array(lst[i]), np.array(lst[j]))

print("\nSimilarity matrix:\n", lst_sim)
