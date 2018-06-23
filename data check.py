import pandas as pd

for file in ['data/book_ratings_train.csv', 'data/book_ratings_test.csv', 'data/books.csv', 'data/implicit_ratings.csv',
             'data/submission.csv', 'data/users.csv']:

    with open(file, encoding='utf-8') as f:
        Train = pd.read_csv(f)

    print(Train.isnull().values.any())

