#NOTEE: If random state is same test set split is subse for smaller test set size in train_test_split
import torch
from torch.autograd import Variable
from sklearn.datasets import load_digits
from sklearn.preprocessing import MultiLabelBinarizer as mlb
from sklearn.preprocessing import LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier as ovr
import sklearn.metrics as met
import numpy as np
import pandas as pd
import pickle
import sys
from sklearn.model_selection import train_test_split

#Hyper parameters
N, r, D_in, H1, H2, D_out = 1797, 0.2, 64, 20, 15, 10
if len(sys.argv) == 1:
    random = 0;
else:
    random = int(sys.argv[1])
    print(random)
#learning_rate = 1e-3
#
##data:
fp = open("MULTILABEL_MNIST.pickle",'rb')
(X,y) = pickle.load(fp)
y = y.reshape(-1,10)
print('average',np.sum(y)/70000)
print(X.shape,y.shape)
#digits = load_digits()
#XX = digits.data
#yy = digits.target
##SVM part :TEMP
#clf = SVC(kernel='linear',decision_function_shape='ovr')
#clf.fit(XX[100:],yy[100:])
#print(clf.score(XX[:100],yy[:100]))
#print(clf.decision_function(XX).shape)
##END TEMP
#lb = LabelBinarizer()
#yy =  lb.fit_transform(yy)
#print(yy.shape,yy[0])
##Train test split
#X, X_t, y, y_t = train_test_split(XX, yy, test_size=r)
#
##SVM part before annotating
#clf = ovr(SVC(kernel='linear'))
#clf.fit(X,y)
#print("SCORE :",clf.score(X_t,y_t))
#print(clf.decision_function(XX).shape)
#print(len(clf.predict(XX)[np.nonzero(np.sum(clf.predict(XX),axis=1)-1)]))
##END
#
##convert to Tensors
#X = Variable(torch.from_numpy(X))
#X = X.type(torch.FloatTensor)
#y = Variable(torch.from_numpy(y))
#y = y.type(torch.FloatTensor)
#    #test set
#X_t = Variable(torch.from_numpy(X_t))
#X_t = X_t.type(torch.FloatTensor)
#y_t = Variable(torch.from_numpy(y_t))
#y_t = y_t.type(torch.FloatTensor)
#
##network to annotate data
#
#model = torch.nn.Sequential(
#        torch.nn.Linear(D_in, H1),
#        torch.nn.LeakyReLU(0.01),
#        #torch.nn.Linear(H1,H2),
#        #torch.nn.ReLU(),
#        torch.nn.Linear(H1,D_out),
#        torch.nn.Softmax(),
#        )
#loss_fn= torch.nn.BCELoss(size_average=True)
#optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
#
#err_train = []
#err_missclass = []
#err_test = []
#err_test_miss = []
##train the network
#for i in range(1,1000):
#    y_pred = model(X_t)   #Test set
#    loss = loss_fn(y_pred, y_t);
#    err_test.append( loss.data[0])
#    miss=np.argmax(y_pred.data.numpy(),axis=1)-np.argmax(y_t.data.numpy(),axis=1)
#    miss=np.count_nonzero(miss)
#    err_test_miss.append(miss)
#
#    y_pred = model(X)     #Train set
#    loss = loss_fn(y_pred, y);
#
#    err_train.append(loss.data[0])
#
#    miss=np.argmax(y_pred.data.numpy(),axis=1)-np.argmax(y.data.numpy(),axis=1)
#    miss=np.count_nonzero(miss)
#    err_missclass.append(miss)
#    print(i,err_train[-1],err_missclass[-1],err_test[-1],err_test_miss[-1])
#
#    optimizer.zero_grad()
#    loss.backward()
#    optimizer.step()

import matplotlib.pyplot as plt

#plt.plot(err_train[10:],'ro',label='Train Error');
#plt.plot(np.asarray(err_missclass[10:])/(N*(1-r)) ,'r+',label='Train Missclassif');
#plt.plot(err_test[10:] ,'go',label='Test Error');
#plt.plot(np.asarray(err_test_miss[10:])/(N*r) ,'g+',label='Test Missclassif');
#plt.yscale('log')
#plt.legend();

#plt.show();

#Probabilities
#for i in range(0,10):
#    y_pred = model(X_t[i:i+1])
#    print(i,'Order :',np.argsort(y_pred[0].data.numpy())[::-1],'actual',np.nonzero(y_t.data[i].numpy())[0][0])
#    print(i, y_pred[0])
#    plt.gray()
#    plt.matshow(X_t.data[i].numpy().reshape(8,8))
#    plt.show()
#

#Relabel the digits to make it more multilabel
#y_pred1 = model(X)
#y_pred2 = model(X_t)
#y_pred1 = y_pred1.data.numpy()
#y_pred2 = y_pred2.data.numpy()
#y_pred1[y_pred1>1e-6]=1
#y_pred1[y_pred1<1]=0
#y_pred2[y_pred2>1e-6]=1
#y_pred2[y_pred2<1]=0
#y_pred1 = y_pred1.astype('int');
#y_pred2 = y_pred2.astype('int');
#print(y_pred1.shape)
#print("Final Error (train, test)      :",err_train[-1],err_test[-1])
#print("Final Miss Classification rate :",err_missclass[-1]/(N*(1-r)),err_test_miss[-1]/(N*r))

#SVM part after annotating
#clf = ovr(SVC(kernel='linear'))
#print(X.data.numpy().shape)
#clf.fit(X.data.numpy(),y_pred1)
#print("SCORE :",clf.score(X_t.data.numpy(),y_pred2))
#print(clf.decision_function(XX).shape)
#print(len(clf.predict(XX)[np.nonzero(np.sum(clf.predict(XX),axis=1)-1)]))
#END

#Function to return best point using our method
def our_method(X,clf,t):
    #Evaluate distance of points to hyperplanes
    dist = clf.decision_function(X)
    #Pick t'th highest value
    dist.sort(axis=1)
    dist1 = dist[:,-t]
    dist2 = dist[:,-(t+1)]
    gap = dist1-dist2
    return np.argmin(gap)

#Function to return closest point
def closest_to_plane(X,clf):
    dist = clf.decision_function(X)
    dist = np.max(dist,axis=1)
    return np.argmin(dist)

#Train SVM using our method
r = 0.998
r2 = 0.995
till = 100
#X_train = X.data.numpy()
#X_test  = X_t.data.numpy()
#X = np.concatenate((X_train,X_test))
#y = np.concatenate((y_pred1,y_pred2))
#print(X.shape,y.shape)

#clf = ovr(LinearSVC())
clf = ovr(SVC(kernel='poly'))
err_rand=[]
err_closest=[]
err_gap = []
print("SPLITTING train and test (1:990)")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=r,random_state=random)
X_test,X_ulabel,y_test,y_ulabel = train_test_split(X_test,y_test, test_size = r2, random_state=random)
print("Done splitting train and test\n Train\t Test\t unlabelled")
print(X_train.shape,X_test.shape,X_ulabel.shape)

for i in range(1,till):
    print(i,"FITTING...")
    clf.fit(X_train, y_train)
    print("Done. Test loss....")
    err_rand.append(clf.score(X_test, y_test))
    print("Done. Picking.....")
    pick = np.random.randint(0,X_ulabel.shape[0])
    idx = np.ones(X_ulabel.shape[0],dtype=bool)
    idx[pick] = False
    X_train = np.insert(X_train,(0),X_ulabel[pick],axis=0)
    y_train = np.insert(y_train,(0),y_ulabel[pick],axis=0)
    X_ulabel= X_ulabel[idx]
    y_ulabel= y_ulabel[idx]
    print("Done.")
    print(i,err_rand[-1],X_ulabel.shape[0])

#print("Closest:")
#print("SPLITTING train and test (1:990)")
#X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=r,random_state=random)
#X_test,X_ulabel,y_test,y_ulabel = train_test_split(X_test,y_test, test_size = r2, random_state=random)
#X_ulabel, y_ulabel = X_ulabel[:1000],y_ulabel[:1000]
#print("Done splitting train and test\n Train\t Test\t unlabelled")
#print(X_train.shape,X_test.shape,X_ulabel.shape)
#for i in range(1,till):
#    print(i,"Fitting...")
#    clf.fit(X_train, y_train)
#    print("Done. Test Error....")
#    err_closest.append(clf.score(X_test, y_test))
#    print("Done. Picking .....")
#
#    pick = closest_to_plane(X_ulabel,clf)
#    idx = np.ones(X_ulabel.shape[0],dtype=bool)
#    idx[pick] = False
#    X_train = np.insert(X_train,(0),X_ulabel[pick],axis=0)
#    y_train = np.insert(y_train,(0),y_ulabel[pick],axis=0)
#    X_ulabel= X_ulabel[idx]
#    y_ulabel= y_ulabel[idx]
#    print("Done")
#    print(i,err_closest[-1], X_ulabel.shape[0])
#
print("Gap based (our method:")
print("SPLITTING train and test (1:990)")
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=r,random_state=random)
X_test,X_ulabel,y_test,y_ulabel = train_test_split(X_test,y_test, test_size = r2, random_state=random)
print("Done splitting train and test\n Train\t Test\t unlabelled")
print(X_train.shape,X_test.shape,X_ulabel.shape)
X_ulabel, y_ulabel = X_ulabel[:1000],y_ulabel[:1000]

for i in range(1,till):
    print("Fitting...")
    clf.fit(X_train, y_train)
    print("Done. Test error....")
    err_gap.append(clf.score(X_test, y_test))
    print("Done. Picking...")

    pick = our_method(X_ulabel,clf,t=1)
    idx = np.ones(X_ulabel.shape[0],dtype=bool)
    idx[pick] = False
    X_train = np.insert(X_train,(0),X_ulabel[pick],axis=0)
    y_train = np.insert(y_train,(0),y_ulabel[pick],axis=0)
    X_ulabel = X_ulabel[idx]
    y_ulabel = y_ulabel[idx]
    print("Done")
    print(i,err_gap[-1], X_ulabel.shape[0])


#Plot actual error
print("Comparing the different elements")
plt.plot(err_rand,'g+',label='Random')
plt.plot(err_closest,'bo',label='Closest')
plt.plot(err_gap,'y^',label='Gap based')
plt.legend()
plt.show()
#Plot improvement
print("Improvement for each iteration:");
improv_rand    = np.diff(err_rand)
improv_closest = np.diff(err_closest)
improv_gap     = np.diff(err_gap)
plt.plot(improv_rand,'g+',label='Random')
plt.plot(improv_closest,'bo',label='Closest')
plt.plot(improv_gap,'y^',label='Gap based')
plt.legend()
plt.show()

print("Number of times Gap performs better than random  :",np.sum(improv_gap>improv_rand))
print("Number of times Gap performs better than Closest :",np.sum(improv_gap>improv_closest))
print("Number of times Gap performs better than both    :",np.sum( (improv_gap>improv_rand) * (improv_gap>improv_closest)))
