import pandas as pd
import implicit
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sparse


# def user_items(u_stars):
#     star_ids = [ISBNId[s] for s in u_stars if s in ISBNId]
#     data = [Confidence for _ in star_ids]
#     rows = [0 for _ in star_ids]
#     shape = (1, Model.item_factors.shape[0])
#     return coo_matrix((data, (rows, star_ids)), shape=shape).tocsr()


with open('user data.csv', encoding='utf-8') as f:
    Data = pd.read_csv(f)
with open('data/book_ratings_test.csv', encoding='utf-8') as f:
    Test = pd.read_csv(f)

Data = Data.dropna(subset=['ISBN'], axis=0)

# User = list(np.sort(Data['User-ID'].unique()))  # Get unique users
# Book = list(Data['ISBN'].unique())  # Get unique books that were rated
# Rate = list(Data['Book-Rating'])  # All of ratings
#
# Rows = Data['User-ID'].astype('category', categories=User).cat.codes
# # Get the associated row indices
# Cols = Data['ISBN'].astype('category', categories=Book).cat.codes
# # Get the associated column indices
# RateSparse = sparse.csr_matrix((Rate, (Rows, Cols)), shape=(len(User), len(Book)))
#
# # MatrixSize = RateSparse.shape[0]*RateSparse.shape[1]  # Number of possible interactions in the matrix
# # RateNum = len(Rate)  # Number of items interacted with
# # Sparsity = 100*(1 - (RateNum/MatrixSize))
# # print(Sparsity)
#
# Alpha = 15
# UserVec, BookVec = implicit.als.AlternatingLeastSquares((RateSparse * Alpha).astype('double'), factors=50,
#                                                         dtype=np.float64, regularization=0.1, iterations=50)


# map each repo and user to a unique numeric value
Data['User-ID'] = Data['User-ID'].astype("category")
Data['ISBN'] = Data['ISBN'].astype("category")
# create a sparse matrix of all the users/repos
Rating = coo_matrix((np.ones(Data.shape[0]),
                    (Data['ISBN'].cat.codes.copy(),
                    Data['User-ID'].cat.codes.copy())))
# train model
Model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, dtype=np.float64, iterations=50)
Confidence = 40
Model.fit(Confidence * Rating)

# ISBN = dict(enumerate(Data['ISBN'].cat.categories))
# ISBNId = {r: i for i, r in ISBN.iteritems()}


user_items = Rating.T.tocsr()
recommendations = Model.recommend('cd5829597b', user_items)
# cd5829597b

