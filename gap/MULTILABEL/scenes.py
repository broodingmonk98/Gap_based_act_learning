from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC,LinearSVC
from sklearn.multiclass import OneVsRestClassifier as ovr


import pickle
import numpy as np
import sys

yeast = fetch_mldata('scene-classification')
X = yeast['data'];
y = yeast['target'].transpose().toarray().astype('int32');
test_size=0.85
pool_size= 0.7
random_state= abs(int(np.random.normal(100,100)))
num = 5
kernel = 'rbf'
X_train, X1, y_train, y1 = train_test_split(X,y,random_state=random_state,test_size=test_size)
X_test, X_pool, y_test, y_pool = train_test_split(X1,y1, random_state=random_state,test_size=pool_size)

if len(sys.argv) == 2:
    random_state = int(sys.argv[1])
if len(sys.argv) == 3:
    random_state = int(sys.argv[1])
    kernel = sys.argv[2]


#fp = open("MULTILABEL_MNIST.pickle",'rb')
#(X,y) = pickle.load(fp)
#print(X.shape,y.shape)
#y = y.reshape(-1,10)

clf = ovr(LinearSVC(C=0.10))
#lb = LabelBinarizer()
#y = lb.fit_transform(y)

#split train and test:
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

err_gap = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],]
print("Gap based picking")
#split train and test again in same manner
t_values = [1,2,3,4,5]
best,bestt = 0,0;
for t in range(1,len(t_values)+1):
    print("Iteration ",t)
    X_train, X1, y_train, y1 = train_test_split(X,y,random_state=random_state,test_size=test_size)
    X_test, X_pool, y_test, y_pool = train_test_split(X1,y1, random_state=random_state,test_size=pool_size)
    for i in range(0,100):
        clf.fit(X_train,y_train)
        err_gap[t].append(clf.score(X_test,y_test))
        print(i,err_gap[t][-1])

        pick = gap_based_method(X_pool,clf,t=t_values[t-1],num=num)
        idx = np.ones(X_pool.shape[0],dtype=bool)
        idx[pick] = False
        X_train = np.insert(X_train,(0),X_pool[pick],axis=0)
        y_train = np.insert(y_train,(0),y_pool[pick],axis=0)
        X_pool= X_pool[idx]
        y_pool= y_pool[idx]
    if sum(err_gap[t][-10:])>best:
        best = sum(err_gap[t][-10:])
        bestt = t;

print('Final Class ratios:')
print('Training set :',np.sum(y_train,axis=0))
print('Test set     :',np.sum(y_test,axis=0))
print('Pool set     :',np.sum(y_pool,axis=0))
print('Train,Test,Pool',y_train.shape[0],y_test.shape[0],y_pool.shape[0])

import matplotlib.pyplot as plt
print("Comparing the different elements")
X_axis = int(X.shape[0]*(1-test_size))+np.arange(0,100)*num
print('BEST:',bestt,best/10);
for i in range(1,len(t_values)+1):
    plt.plot(X_axis,err_rand,'g+',label='Random')
#plt.plot(err_closest,'bo',label='Closest')
    plt.plot(X_axis,err_gap[i],'r1',label='Gap based'+str(t_values[i-1]))

#plt.plot(X_axis,err_gap[2],'b2',label='Gap based'+str(t_values[1]))
#plt.plot(X_axis,err_gap[3],'y3',label='Gap based'+str(t_values[2]))
#plt.plot(X_axis,err_gap[4],'p^',label='Gap based4')
#plt.plot(X_axis,err_gap,'c^',label='Gap based5')
    plt.xlabel('Points in training set')
    plt.ylabel('Score')
    plt.legend()
    plt.show()
