# u.user 파일을 DataFrame으로 읽기 
import pandas as pd
u_cols = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
users = pd.read_csv('data/u.user', sep='|', names=u_cols, encoding='latin-1')
users = users.set_index('user_id')
print(users.head())
'''
         age sex  occupation zip_code
user_id
1         24   M  technician    85711
2         53   F       other    94043
3         23   M      writer    32067
4         24   M  technician    43537
5         33   F       other    15213
'''

# u.item 파일을 DataFrame으로 읽기
import pandas as pd
i_cols = ['movie_id', 'title', 'release date', 'video release date', 'IMDB URL', 
          'unknown', 'Action', 'Adventure', 'Animation', 'Children\'s', 'Comedy', 
          'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 
          'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
movies = pd.read_csv('data/u.item', sep='|', names=i_cols, encoding='latin-1')
movies = movies.set_index('movie_id')
print(movies.head())
'''
                      title release date  video release date                                           IMDB URL  ...  Sci-Fi  Thriller  War  Western
movie_id                                                                                                         ...
1          Toy Story (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Toy%20Story%2...  ...       0         0    0        0       
2          GoldenEye (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?GoldenEye%20(...  ...       0         1    0        0       
3         Four Rooms (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Four%20Rooms%...  ...       0         1    0        0       
4         Get Shorty (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Get%20Shorty%...  ...       0         0    0        0       
5            Copycat (1995)  01-Jan-1995                 NaN  http://us.imdb.com/M/title-exact?Copycat%20(1995)  ...       0         1    0        0       

[5 rows x 23 columns]
'''

# u.data 파일을 DataFrame으로 읽기
import pandas as pd
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/u.data', sep='\t', names=r_cols, encoding='latin-1') 
ratings = ratings.set_index('user_id')
print(ratings.head())
'''
         movie_id  rating  timestamp
user_id
196           242       3  881250949
186           302       3  891717742
22            377       1  878887116
244            51       2  880606923
166           346       1  886397596
'''

# Best-seller 추천 
def recom_movie1(n_items):
    movie_sort = movie_mean.sort_values(ascending=False)[:n_items]
    recom_movies = movies.loc[movie_sort.index]
    recommendations = recom_movies['title']
    return recommendations

movie_mean = ratings.groupby(['movie_id'])['rating'].mean()
print(recom_movie1(5))
'''
movie_id
1293                                      Star Kid (1997)
1467                 Saint of Fort Washington, The (1993)
1653    Entertaining Angels: The Dorothy Day Story (1996)
814                         Great Day in Harlem, A (1994)
1122                       They Made Me a Criminal (1939)
Name: title, dtype: object
'''

def recom_movie2(n_items):
   return movies.loc[movie_mean.sort_values(ascending=False)[:n_items].index]['title']

print(recom_movie2(5))
'''
movie_id
1293                                      Star Kid (1997)
1467                 Saint of Fort Washington, The (1993)
1653    Entertaining Angels: The Dorothy Day Story (1996)
814                         Great Day in Harlem, A (1994)
1122                       They Made Me a Criminal (1939)
Name: title, dtype: object
'''

# 정확도 계산 
import numpy as np

def RMSE(y_true, y_pred):
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

rmse = []
for user in set(ratings.index):
    y_true = ratings.loc[user]['rating']
    y_pred = movie_mean[ratings.loc[user]['movie_id']]
    accuracy = RMSE(y_true, y_pred)
    rmse.append(accuracy)

print(np.mean(rmse))
# 0.996007224010567
