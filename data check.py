import numpy as np
from numpy import genfromtxt
import pandas as pd

for file in ['data/book_ratings_train.csv', 'data/book_ratings_teat.csv', 'books', 'implicit_ratings.csv',
             'submission.csv', 'users.csv']:

    with open('data/book_ratings_train.csv', encoding='utf-8') as f:
        Train = pd.read_csv(f)

    print(Train.isnull().values.any())

