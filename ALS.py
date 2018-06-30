import pandas as pd
import implicit
import numpy as np
from scipy.sparse import coo_matrix
import random

Try = 19

with open('user data.csv', encoding='utf-8') as f:
    Data = pd.read_csv(f)
with open('data/book_ratings_test.csv', encoding='utf-8') as f:
    Test = pd.read_csv(f)

Data = Data.dropna(subset=['ISBN'], axis=0)

# map each repo and user to a unique numeric value
Data['User-ID'] = Data['User-ID'].astype("category")
Data['ISBN'] = Data['ISBN'].astype("category")
# create a sparse matrix of all the users/repos
Rating = coo_matrix((np.ones(Data.shape[0]),
                    (Data['ISBN'].cat.codes.copy(),
                    Data['User-ID'].cat.codes.copy())))
# train model
Model = implicit.als.AlternatingLeastSquares(factors=1, regularization=0, dtype=np.float64, iterations=1)
Confidence = 10000
Mag = 2
# 1, 0.01, 1, 10000, 2 -> 1.68 / 29.26 (try18)
Model.fit(Confidence * Rating)

ISBN = dict(enumerate(Data['ISBN'].cat.categories))
ISBNId = {r: i for i, r in ISBN.items()}  # find numbers with id

User = dict(enumerate(Data['User-ID'].cat.categories))
UserId = {r: i for i, r in User.items()}

Predictions = []
j = 0

for i, k in zip(Test['User-ID'], Test['ISBN']):
    try:
        UserVec = Model.user_factors[UserId['%s' % i]]
        # BookVec = Model.item_factors[ISBNId['%s' % k]]
        # PredictValue = np.dot(UserVec, BookVec.T)
        PredictVec = np.dot(UserVec, Model.item_factors.T)
        PredictVec = PredictVec * (10/PredictVec.max()) * Mag  # normalize
        PredictValue = PredictVec[ISBNId['%s' % k]]  # find the exact value
        if PredictValue >= 10:
            PredictValue = 10
        if PredictValue < 1:
            PredictValue = 1
        Predictions.append(PredictValue)
    except KeyError:
        # Predictions.append(random.randint(1, 10))
        Predictions.append(8)
        # Predictions.append(None)
    except IndexError:
        # Predictions.append(random.randint(1, 10))
        Predictions.append(8)
        # Predictions.append(None)
    j = j + 1

    # print(i)
    # print(k)
    print('round: %s , prediction: %s' % (j, Predictions[-1]))

Output = pd.DataFrame(Predictions, columns=['Rating'])
Output_ = pd.DataFrame([None]*len(Predictions), columns=['Rating'])

# Output['Rating'] = Output['Rating'].fillna(Output['Rating'].mean())
Output.to_csv('try%s_float.csv' % Try, index=False, header=False)

Output_['Rating'] = Output['Rating'].apply(np.round)
Output_['Rating'] = Output_['Rating'].astype('int64')
Output_.to_csv('try%s_int.csv' % Try, index=False, header=False)
