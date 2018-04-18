from sklearn.datasets import load_digits
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as ovr
import pickle
import numpy as np
import sys

test_size=0.95
pool_size= 0.7
random_state=0
num = 5
kernel = 'poly'

if len(sys.argv) == 2:
    random_state = int(sys.argv[1])
if len(sys.argv) == 3:
    random_state = int(sys.argv[1])
    kernel = sys.argv[2]


#fp = open("MULTILABEL_MNIST.pickle",'rb')
#(X,y) = pickle.load(fp)
#print(X.shape,y.shape)
#y = y.reshape(-1,10)

clf = ovr(SVC(kernel=kernel))
digits = load_digits()
X = digits.data
y = digits.target
lb = LabelBinarizer()
y = lb.fit_transform(y)

#split train and test:
X_train, X1, y_train, y1 = train_test_split(X,y,random_state=random_state,test_size=test_size)
X_test, X_pool, y_test, y_pool = train_test_split(X1,y1, random_state=random_state,test_size=pool_size)
print('Class ratios :')
print('Training set :',np.sum(y_train,axis=0))
print('Test set     :',np.sum(y_test,axis=0))
print('Pool set     :',np.sum(y_pool,axis=0))
print('Train,Test,Pool',y_train.shape[0],y_test.shape[0],y_pool.shape[0])
#TODO class ratio preserving split
def gap_based_method(X,clf,t,num):
    #Evaluate distance of points to hyperplanes
    dist = clf.decision_function(X)
    #Pick t'th highest value
    dist.sort(axis=1)
    dist1 = dist[:,-t]
    dist2 = dist[:,-(t+1)]
    gap = dist1-dist2
    return np.argsort(gap)[:num]

#evaluate based no random picking
err_rand = []
print("Random picking")
for i in range(0,100):
    clf.fit(X_train,y_train)
    err_rand.append(clf.score(X_test,y_test))
    print(i,err_rand[-1])

    pick = np.random.randint(0,X_pool.shape[0],num)
    idx = np.ones(X_pool.shape[0],dtype=bool)
    idx[pick] = False
    X_train = np.insert(X_train,(0),X_pool[pick],axis=0)
    y_train = np.insert(y_train,(0),y_pool[pick],axis=0)
    X_pool= X_pool[idx]
    y_pool= y_pool[idx]

err_gap = [[],[],[],[],[]]
print("Gap based picking")
#split train and test again in same manner

for t in range(1,4):
    print("Iteration ",t)
    X_train, X1, y_train, y1 = train_test_split(X,y,random_state=random_state,test_size=test_size)
    X_test, X_pool, y_test, y_pool = train_test_split(X1,y1, random_state=random_state,test_size=pool_size)
    for i in range(0,100):
        clf.fit(X_train,y_train)
        err_gap[t].append(clf.score(X_test,y_test))
        print(i,err_gap[t][-1])

        pick = gap_based_method(X_pool,clf,t=t,num=num)
        idx = np.ones(X_pool.shape[0],dtype=bool)
        idx[pick] = False
        X_train = np.insert(X_train,(0),X_pool[pick],axis=0)
        y_train = np.insert(y_train,(0),y_pool[pick],axis=0)
        X_pool= X_pool[idx]
        y_pool= y_pool[idx]

print('Final Class ratios:')
print('Training set :',np.sum(y_train,axis=0))
print('Test set     :',np.sum(y_test,axis=0))
print('Pool set     :',np.sum(y_pool,axis=0))
print('Train,Test,Pool',y_train.shape[0],y_test.shape[0],y_pool.shape[0])

import matplotlib.pyplot as plt
print("Comparing the different elements")
X_axis = int(X.shape[0]*(1-test_size))+np.arange(0,100)*num
plt.plot(X_axis,err_rand,'g+',label='Random')
#plt.plot(err_closest,'bo',label='Closest')
plt.plot(X_axis,err_gap[1],'y1',label='Gap based1')
plt.plot(X_axis,err_gap[2],'b2',label='Gap based2')
plt.plot(X_axis,err_gap[3],'r3',label='Gap based3')
#plt.plot(X_axis,err_gap[4],'p^',label='Gap based4')
#plt.plot(X_axis,err_gap,'c^',label='Gap based5')
plt.xlabel('Points in training set')
plt.ylabel('Score')
plt.legend()
plt.show()
