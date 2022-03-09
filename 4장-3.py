import numpy as np
import pandas as pd

r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_csv('data/u.data', names=r_cols,  sep='\t',encoding='latin-1')
ratings = ratings[['user_id', 'movie_id', 'rating']].astype(int)            # timestamp 제거

##### (1)

# train test 분리
from sklearn.utils import shuffle
TRAIN_SIZE = 0.75
ratings = shuffle(ratings, random_state=1)
cutoff = int(TRAIN_SIZE * len(ratings))
ratings_train = ratings.iloc[:cutoff]
ratings_test = ratings.iloc[cutoff:]
    
# New MF class for training & testing
class NEW_MF():
    def __init__(self, ratings, K, alpha, beta, iterations, verbose=True):
        self.R = np.array(ratings)
##### >>>>> (2) user_id, item_id를 R의 index와 매핑하기 위한 dictionary 생성
        item_id_index = []
        index_item_id = []
        for i, one_id in enumerate(ratings):
            item_id_index.append([one_id, i])
            index_item_id.append([i, one_id])
        self.item_id_index = dict(item_id_index)
        self.index_item_id = dict(index_item_id)        
        user_id_index = []
        index_user_id = []
        for i, one_id in enumerate(ratings.T):
            user_id_index.append([one_id, i])
            index_user_id.append([i, one_id])
        self.user_id_index = dict(user_id_index)
        self.index_user_id = dict(index_user_id)
#### <<<<< (2)
        self.num_users, self.num_items = np.shape(self.R)
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.iterations = iterations
        self.verbose = verbose

    # train set의 RMSE 계산
    def rmse(self):
        xs, ys = self.R.nonzero()
        self.predictions = []
        self.errors = []
        for x, y in zip(xs, ys):
            prediction = self.get_prediction(x, y)
            self.predictions.append(prediction)
            self.errors.append(self.R[x, y] - prediction)
        self.predictions = np.array(self.predictions)
        self.errors = np.array(self.errors)
        return np.sqrt(np.mean(self.errors**2))

    # Ratings for user i and item j
    def get_prediction(self, i, j):
        prediction = self.b + self.b_u[i] + self.b_d[j] + self.P[i, :].dot(self.Q[j, :].T)
        return prediction

    # Stochastic gradient descent to get optimized P and Q matrix
    def sgd(self):
        for i, j, r in self.samples:
            prediction = self.get_prediction(i, j)
            e = (r - prediction)

            self.b_u[i] += self.alpha * (e - self.beta * self.b_u[i])
            self.b_d[j] += self.alpha * (e - self.beta * self.b_d[j])

            self.P[i, :] += self.alpha * (e * self.Q[j, :] - self.beta * self.P[i,:])
            self.Q[j, :] += self.alpha * (e * self.P[i, :] - self.beta * self.Q[j,:])

##### >>>>> (3)
    # Test set을 선정
    def set_test(self, ratings_test):
        test_set = []
        for i in range(len(ratings_test)):      # test 데이터에 있는 각 데이터에 대해서
            x = self.user_id_index[ratings_test.iloc[i, 0]]
            y = self.item_id_index[ratings_test.iloc[i, 1]]
            z = ratings_test.iloc[i, 2]
            test_set.append([x, y, z])
            self.R[x, y] = 0                    # Setting test set ratings to 0
        self.test_set = test_set
        return test_set                         # Return test set

    # Test set의 RMSE 계산
    def test_rmse(self):
        error = 0
        for one_set in self.test_set:
            predicted = self.get_prediction(one_set[0], one_set[1])
            error += pow(one_set[2] - predicted, 2)
        return np.sqrt(error/len(self.test_set))

    # Training 하면서 test set의 정확도를 계산
    def test(self):
        # Initializing user-feature and item-feature matrix
        self.P = np.random.normal(scale=1./self.K, size=(self.num_users, self.K))
        self.Q = np.random.normal(scale=1./self.K, size=(self.num_items, self.K))

        # Initializing the bias terms
        self.b_u = np.zeros(self.num_users)
        self.b_d = np.zeros(self.num_items)
        self.b = np.mean(self.R[self.R.nonzero()])

        # List of training samples
        rows, columns = self.R.nonzero()
        self.samples = [(i, j, self.R[i,j]) for i, j in zip(rows, columns)]

        # Stochastic gradient descent for given number of iterations
        training_process = []
        for i in range(self.iterations):
            np.random.shuffle(self.samples)
            self.sgd()
            rmse1 = self.rmse()
            rmse2 = self.test_rmse()
            training_process.append((i+1, rmse1, rmse2))
            if self.verbose:
                if (i+1) % 10 == 0:
                    print("Iteration: %d ; Train RMSE = %.4f ; Test RMSE = %.4f" % (i+1, rmse1, rmse2))
        return training_process

    # Ratings for given user_id and item_id
    def get_one_prediction(self, user_id, item_id):
        return self.get_prediction(self.user_id_index[user_id], self.item_id_index[item_id])

    # Full user-movie rating matrix
    def full_prediction(self):
        return self.b + self.b_u[:,np.newaxis] + self.b_d[np.newaxis,:] + self.P.dot(self.Q.T)

##### <<<<< (3)

# 최적의 K값 찾기
results = []
index = []
for K in range(50, 261, 10):
    print('K =', K)
    R_temp = ratings.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    mf = NEW_MF(R_temp, K=K, alpha=0.001, beta=0.02, iterations=300, verbose=True)
    test_set = mf.set_test(ratings_test)
    result = mf.test()
    index.append(K)
    results.append(result)
'''
...(생략)...
K = 240
Iteration: 10 ; Train RMSE = 0.9664 ; Test RMSE = 0.9834
Iteration: 20 ; Train RMSE = 0.9420 ; Test RMSE = 0.9644
Iteration: 30 ; Train RMSE = 0.9314 ; Test RMSE = 0.9566
Iteration: 40 ; Train RMSE = 0.9253 ; Test RMSE = 0.9524
Iteration: 50 ; Train RMSE = 0.9215 ; Test RMSE = 0.9497
Iteration: 60 ; Train RMSE = 0.9188 ; Test RMSE = 0.9480
Iteration: 70 ; Train RMSE = 0.9167 ; Test RMSE = 0.9468
Iteration: 80 ; Train RMSE = 0.9150 ; Test RMSE = 0.9459
Iteration: 90 ; Train RMSE = 0.9134 ; Test RMSE = 0.9452
Iteration: 100 ; Train RMSE = 0.9117 ; Test RMSE = 0.9445
Iteration: 110 ; Train RMSE = 0.9097 ; Test RMSE = 0.9437
Iteration: 120 ; Train RMSE = 0.9070 ; Test RMSE = 0.9427
Iteration: 130 ; Train RMSE = 0.9031 ; Test RMSE = 0.9412
Iteration: 140 ; Train RMSE = 0.8975 ; Test RMSE = 0.9390
Iteration: 150 ; Train RMSE = 0.8898 ; Test RMSE = 0.9360
Iteration: 160 ; Train RMSE = 0.8800 ; Test RMSE = 0.9324
Iteration: 170 ; Train RMSE = 0.8686 ; Test RMSE = 0.9286
Iteration: 180 ; Train RMSE = 0.8557 ; Test RMSE = 0.9250
Iteration: 190 ; Train RMSE = 0.8416 ; Test RMSE = 0.9216
Iteration: 200 ; Train RMSE = 0.8260 ; Test RMSE = 0.9185
Iteration: 210 ; Train RMSE = 0.8089 ; Test RMSE = 0.9158
Iteration: 220 ; Train RMSE = 0.7901 ; Test RMSE = 0.9135
Iteration: 230 ; Train RMSE = 0.7698 ; Test RMSE = 0.9117
Iteration: 240 ; Train RMSE = 0.7482 ; Test RMSE = 0.9103
Iteration: 250 ; Train RMSE = 0.7255 ; Test RMSE = 0.9096
Iteration: 260 ; Train RMSE = 0.7020 ; Test RMSE = 0.9094
Iteration: 270 ; Train RMSE = 0.6780 ; Test RMSE = 0.9098
Iteration: 280 ; Train RMSE = 0.6538 ; Test RMSE = 0.9106
Iteration: 290 ; Train RMSE = 0.6297 ; Test RMSE = 0.9118
Iteration: 300 ; Train RMSE = 0.6059 ; Test RMSE = 0.9132
...(생략)...
'''

# 최적의 iterations 값 찾기
summary = []
for i in range(len(results)):
    RMSE = []
    for result in results[i]:
        RMSE.append(result[2])
    min = np.min(RMSE)
    j = RMSE.index(min)
    summary.append([index[i], j+1, RMSE[j]])

# 그래프 그리기
import matplotlib.pyplot as plt
plt.plot(index, [x[2] for x in summary])
plt.ylim(0.89, 0.94)
plt.xlabel('K')
plt.ylabel('RMSE')
plt.show()
