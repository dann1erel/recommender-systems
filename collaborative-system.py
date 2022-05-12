import numpy as np
import pandas as pd
from config import settings

ratings_df = pd.read_csv(settings['collab_file'])

n = 5000
ratings_df_sample = ratings_df[:n]
user_ratings = [5.0, 0.0, 3.0, 0.0, 3.0]

n_users = len(ratings_df_sample['userId'].unique())
n_movies = len(ratings_df_sample['movieId'].unique())

print('Unique users count: {}'.format(len(ratings_df_sample['userId'].unique())))
print('Unique movies count: {}'.format(len(ratings_df_sample['movieId'].unique())))
print('DataFrame shape: {}'.format(ratings_df_sample.shape))
print("# {}".format(ratings_df.head(n=15)))

lst = np.array([[0.0]*ratings_df_sample['movieId'].max()]*len(ratings_df_sample['userId'].unique()))

for x in range(ratings_df_sample.shape[0]):
    lst[tuple(ratings_df_sample['userId'])[x]-1][tuple(ratings_df_sample['movieId'])[x]-1]\
        = tuple(ratings_df_sample['rating'])[x]

print("\nRatings:")
print(*lst, sep='\n')
print("\nUser ratings: ", user_ratings)


def cos_similarity(vec_a, vec_b):
    # dot - scalar product, norm - vector length???
    result = (np.dot(vec_a, vec_b))/(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    return result


lst_sim = np.array([0.0]*len(ratings_df_sample['userId'].unique()))

for i in range(len(ratings_df_sample['userId'].unique())):
    lst_sim[i] = cos_similarity(user_ratings, np.array(lst[i]))

print("\nSimilarity matrix:\n", lst_sim)

lst_sim_enum = tuple(sorted(enumerate(lst_sim), key=lambda el: el[1], reverse=True))

print(lst_sim_enum)

sim_border = 0.6

films_to_rec = [i for i in range(len(user_ratings)) if user_ratings[i] == 0]

for i in films_to_rec:
    ratings_sum = 0
    vectors_amount = 0
    vect_check = 0
    for ind, dist in lst_sim_enum:
        if dist > sim_border:
            vect_check = lst[ind]
            ratings_sum += vect_check[i]
            vectors_amount += 1 if vect_check[i] != 0 else 0
    user_ratings[i] = ratings_sum / vectors_amount if vect_check[i] != 0 else 0.0

print(user_ratings)
