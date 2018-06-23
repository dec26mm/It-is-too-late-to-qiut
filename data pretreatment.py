import pandas as pd

with open('data/users.csv', encoding='utf-8') as f:
    User = pd.read_csv(f)

with open('data/implicit_ratings.csv', encoding='utf-8') as f:
    ZeroRating = pd.read_csv(f)

with open('data/book_ratings_train.csv', encoding='utf-8') as f:
    Train = pd.read_csv(f)

ZeroRating['Book-Rating'] = 1
User['Age'] = User['Age'].fillna(User['Age'].mean())
Train = pd.concat([Train, ZeroRating], keys=['User-ID', 'ISBN', 'Book-Rating'], axis=0)

Data = pd.merge(Train, User, on=['User-ID'], how='outer')

Data.to_csv('user data.csv', encoding='utf-8')




