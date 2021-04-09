from surprise import KNNBasic,SVD
from surprise import Reader, Dataset
from surprise.model_selection import cross_validate
import pandas as pd
import numpy as np


#Read data from “ratings.csv” with line format: 'userID movieID rating timestamp' and get rid of 'timestamp'
ratings = pd.read_csv('ratings_small.csv')
reader = Reader()
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

algo_item_based_MSD = KNNBasic(sim_options={'user_based': False})
algo_user_based_MSD = KNNBasic()
algo_pmf = SVD()

algo_item_based_cosine = KNNBasic(sim_options={'name':'cosine','user_based': False})
algo_user_based_cosine = KNNBasic(sim_options={'name':'cosine'})

algo_item_based_ps = KNNBasic(sim_options={'name': 'pearson','user_based': False})
algo_user_based_ps = KNNBasic(sim_options={'name': 'pearson'})


#Compute the average MAE and RMSE 
print("The item based:")
cross_validate(algo_item_based_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("The user based:")
cross_validate(algo_user_based_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("PMF:")
cross_validate(algo_pmf, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)


#Examine how the cosine, MSD (Mean Squared Difference), and Pearson similarities impact the performances of User based Collaborative Filtering and Item based Collaborative Filtering. Plot your results. Is the impact of the three metrics on User based Collaborative Filtering consistent with the impact of the three metrics on Item based Collaborative Filtering?
#the cosine, MSD (Mean Squared Difference), and Pearson similarities  for item based
print("item based MSD:")
cross_validate(algo_item_based_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("item based cosine:")
cross_validate(algo_item_based_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("item based pearson similarity:")
cross_validate(algo_item_based_ps, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

#the cosine, MSD (Mean Squared Difference), and Pearson similarities  for user based
print("user based MSD:")
cross_validate(algo_user_based_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("user based cosine:")
cross_validate(algo_user_based_cosine, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
print("user based pearson similarity:")
cross_validate(algo_user_based_ps, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)



k_array=[10,20,40,100]
for i in k_array:
    algo_item_based_MSD = KNNBasic(k=i,sim_options={'user_based': False})
    algo_user_based_MSD = KNNBasic(k=i)
    print("item based with k=",i)
    cross_validate(algo_item_based_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print("user based with k=",i)
    cross_validate(algo_user_based_MSD, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

    
    
