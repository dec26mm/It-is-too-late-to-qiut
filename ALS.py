import pandas as pd
import implicit
import numpy as np
from scipy.sparse import coo_matrix

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
Model = implicit.als.AlternatingLeastSquares(factors=50, regularization=0.01, dtype=np.float64, iterations=50)
Confidence = 4000
Model.fit(Confidence * Rating)

ISBN = dict(enumerate(Data['ISBN'].cat.categories))
ISBNId = {r: i for i, r in ISBN.items()}  # find numbers with id

User = dict(enumerate(Data['User-ID'].cat.categories))
UserId = {r: i for i, r in User.items()}

Predictions = []

for i, k in zip(Test['User-ID'], Test['ISBN']):
    try:
        UserVec = Model.user_factors[UserId['%s' % i]]
        BookVec = Model.item_factors[ISBNId['%s' % k]]
        PredictValue = np.dot(BookVec, UserVec.T)
        if PredictValue >= 10:
            PredictValue = 10
        if PredictValue < 1:
                PredictValue = 1
        Predictions.append(PredictValue)
    except KeyError:
        Predictions.append(5)  # the average of 1-10
    except IndexError:
        Predictions.append(5)  # the average of 1-10

    print(i)
    print(k)
    print(Predictions[-1])

Output = pd.DataFrame(Predictions, columns=['Rating'])

Output.to_csv('try 2.csv', index=False, header=False)


