# Time Series and Ada Boost

import numpy as np
from sklearn.metrics import classification_report
import time

history = np.genfromtxt("/Users/linzhang/Documents/LC5_Assessment/history.csv", delimiter=",", dtype=None, names=True)
n = history.shape[0]


his = dict()
times = dict()
for i in range(n):
    series = list(history[i, ])[:25]  # not to select the last column
    id = series[0]
    time = series[1].replace('"', '') + '_' + series[2].replace('"', '')
    if id not in his.keys():
        times[id] = 1
    else:
        times[id] += 1
    series.append(times[id])
    start, end=series[3:5]
    if int(end[8])-int(start[8])==0:
        duration = int(end[-7:-1])-int(start[-7:-1])
    else:
        duration = int(end[-7:-1])-int(start[-7:-1])+240000
    series.append(duration)
    if id not in his.keys():
        his[id] = list(series[5:])
    else:
        his[id] = his[id] + (series[5:])

# average the features

for id in his.keys():
    total = len(his[id])
    tmp = np.array(his[id])
    tmp = tmp.reshape(total/22, 22)
    his[id] = np.sum(tmp, axis=0).tolist()

data = np.genfromtxt("Users/linzhang/Documents/LC5_Assessment/train.csv", delimiter=",",dtype=None, names=True)

data = [x.tolist() for x in data]
data = np.array(data)
ids = list(data[:, 0])

for id in ids:
    idx = ids.index(id)
    his[id].append(int(data[idx, 1]))


def DTWDistance(s1, s2, w):
    DTW = {}

    w = max(w, abs(len(s1) - len(s2)))

    for i in range(-1, len(s1)):
        for j in range(-1, len(s2)):
            DTW[(i, j)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(max(0, i - w), min(len(s2), i + w)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return np.sqrt(DTW[len(s1) - 1, len(s2) - 1])


def LB_Keogh(s1, s2, r):
    LB_sum = 0
    for ind, i in enumerate(s1):

        lower_bound = min(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])
        upper_bound = max(s2[(ind - r if ind - r >= 0 else 0):(ind + r)])

        if i > upper_bound:
            LB_sum += (i - upper_bound) ** 2
        elif i < lower_bound:
            LB_sum += (i - lower_bound) ** 2

    return np.sqrt(LB_sum)


def knn_cv(his, w, K):
    preds = []
    y_test = []

    for k in xrange(K):
        # train test split
        training_idx = [i for i in indices if i % K != k]
        test_idx = [i for i in indices if i % K == k]

        train_ids = [ids[x] for x in training_idx]
        test_ids = [ids[x] for x in test_idx]
        train = [his[x] for x in train_ids]
        train = np.array(train)
        test = [his[x] for x in test_ids]
        test = np.array(test)

        for ind, i in enumerate(test):
            y_test.append(i[-1])
            min_dist = float('inf')
            closest_seq = []
            # print ind
            for j in train:
                if LB_Keogh(i[:-1], j[:-1], 5) < min_dist:
                    dist = DTWDistance(i[:-1], j[:-1], w)
                    if dist < min_dist:
                        min_dist = dist
                        closest_seq = j
            preds.append(closest_seq[-1])
    return classification_report(y_test, preds)

start = time.time()
out = knn_cv(his, w=4, K=5)
end = time.time()
print out
print(end - start)



### svm

from sklearn import svm
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix

X=[his[x][:-1] for x in ids]
X=np.array(X)
y=[his[x][-1] for x in ids]
y=np.array(y)
start = time.time()
clf_lin = svm.SVC()
svmpredicted = cross_val_predict(clf_lin, X, y, cv=5)
print(accuracy_score(y, svmpredicted))
end = time.time()
print(end - start)

import matplotlib.pyplot as plt
from collections import Counter
plt.imshow(np.log(confusion_matrix(y, svmpredicted)+1),
           cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()


### logistic regression

from sklearn.linear_model import LogisticRegression
start = time.time()
logreg = LogisticRegression()
lrpredicted = cross_val_predict(logreg, X, y, cv=5)
print(accuracy_score(y, lrpredicted))
end = time.time()
print(end - start)

plt.imshow(np.log(confusion_matrix(y, lrpredicted)+1),
           cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()

### default knn

from sklearn.neighbors import KNeighborsClassifier
def knn_cv(idx, num):
    test_errors = []

    for k in xrange(5):
        # train test split
        training_idx = [i for i in idx if i % 5 != k]
        test_idx = [i for i in idx if i % 5 == k]

        train_ids = [ids[x] for x in training_idx]
        test_ids = [ids[x] for x in test_idx]
        trainset = [his[x] for x in train_ids]
        trainset = np.array(trainset)
        testset = [his[x] for x in test_ids]
        testset = np.array(testset)

        neigh = KNeighborsClassifier(n_neighbors=num)
        neigh.fit(trainset[:, :-1], trainset[:, -1])


#        for real_test_predict, discrete_train_predict in zip(
#                bdt_real.staged_predict(testset[:, :-1]), bdt_discrete.staged_predict(testset[:, :-1])):
        y_pred=neigh.predict(testset[:, :-1])
        test_errors.append(1-accuracy_score(y_pred, testset[:, -1]))
    return test_errors

indices = np.random.permutation(data.shape[0])
start = time.time()
out=knn_cv(indices, num=1)
end = time.time()
print(end - start)


### adaboost
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

def ada(indicies):
    discrete_test_errors = []
#    real_test_errors = []

    for k in xrange(5):
        # train test split
        training_idx = [i for i in indices if i % 5 != k]
        test_idx = [i for i in indices if i % 5 == k]

        train_ids = [ids[x] for x in training_idx]
        test_ids = [ids[x] for x in test_idx]
        trainset = [his[x] for x in train_ids]
        trainset = np.array(trainset)
        testset = [his[x] for x in test_ids]
        testset = np.array(testset)

#        bdt_real = AdaBoostClassifier(
#            DecisionTreeClassifier(max_depth=2),
#            n_estimators=500,
#            learning_rate=1)
        bdt_discrete = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2),
            n_estimators=500,
            learning_rate=1.5,
            algorithm="SAMME")

#        bdt_real.fit(trainset[:, :-1], trainset[:, -1])
        bdt_discrete.fit(trainset[:, :-1], trainset[:, -1])

#        for real_test_predict, discrete_train_predict in zip(
#                bdt_real.staged_predict(testset[:, :-1]), bdt_discrete.staged_predict(testset[:, :-1])):
        pred_dis=bdt_discrete.predict(testset[:, :-1])
        discrete_test_errors.append(1-accuracy_score(pred_dis, testset[:, -1]))
    return discrete_test_errors

indices = np.random.permutation(data.shape[0])
start = time.time()
out=ada(indices)
end = time.time()
print(end - start)


def ada_pred(indicies):
    test_pred = []
    y_test = []
    for k in xrange(5):
        # train test split
        training_idx = [i for i in indices if i % 5 != k]
        test_idx = [i for i in indices if i % 5 == k]

        train_ids = [ids[x] for x in training_idx]
        test_ids = [ids[x] for x in test_idx]
        trainset = [his[x] for x in train_ids]
        trainset = np.array(trainset)
        testset = [his[x] for x in test_ids]
        testset = np.array(testset)

        bdt_discrete = AdaBoostClassifier(
            DecisionTreeClassifier(max_depth=2),
            n_estimators=500,
            learning_rate=1.5,
            algorithm="SAMME")

        bdt_discrete.fit(trainset[:, :-1], trainset[:, -1])
        test_pred.append(bdt_discrete.predict(testset[:, :-1]).tolist())
        y_test.append(testset[:, -1].tolist())
    return [np.array(test_pred), np.array(y_test)]

indices = np.random.permutation(data.shape[0])
res = ada_pred(indices)
adapredicted = res[0]
y_test = res[1]


accuracy_score(adapredicted.flatten(), y_test.flatten())


plt.imshow(np.log(confusion_matrix(y_test.flatten(), adapredicted.flatten())+1),
           cmap='Blues', interpolation='nearest')
plt.grid(False)
plt.ylabel('true')
plt.xlabel('predicted')
plt.show()


