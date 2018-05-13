import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

t = 3
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        #download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
#testloader = torch.utils.data.DataLoader(testset, batch_size=4,
#                                         shuffle=False, num_workers=2)

#classes = ('plane', 'car', 'bird', 'cat',
#           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
print("DONE downloading");
X_test,y_test = [],[]
for i in testset:
    (X,y) = i
    X_test.append(X.numpy())
    y_test.append(y)
X_test = np.asarray(X_test)
X_test = X_test.reshape(-1,32*32*3)
y_test = np.asarray(y_test)
print("Done with test set")

#X_train,y_train = [],[]
#for i in trainset:
#    (X,y) = i
#    X_train.append(X.numpy())
#    y_train.append(y)
#X_train = np.asarray(X_train)
#print(X_train.shape)
#X_train= X_train.reshape(-1,32*32*3)
#y_train = np.asarray(y_train)
#print("Done with train set")

#train our classifer
from sklearn.svm import SVC
from sklearn import cross_validation,metrics

clf = SVC(kernel='poly',decision_function_shape='ovr');
clf.fit(X_test[:300],y_test[:300])
err = clf.score(X_test[-100:],y_test[-100:])
print(err)
#
#def ClosestToLine(clf,points,number,alreadyThere,t):
#    """Returns 'number' of points closest to the hyperplane"""
###nnz data indices indptr has_sortedindices
#    dist = abs(clf._decision_function(points))
#    print("Initial DIST.SHAPE :"+str(dist.shape));
#    dist = np.diff(dist)
#    dist.sort()
#    dist = dist[:,t]
#    print("Final DIST.SHAPE :"+str(dist.shape));
#
#    print("DIST.SHAPE :"+str(dist.shape));
#    mask = np.zeros(dist.shape, dtype=bool);
#    mask[alreadyThere] = True;
#    dist[mask] = float('inf');
#    return dist.argsort()[:number];
#
##Modify data to get required format
#
#temp = [];
#print("ONE")
#for i in trainset.train_data:  #convert to suitable form
#    temp.append(i.reshape(1,-1)[0])
#temp = np.asarray(temp)
#X_train, X_test, y_train, y_test = cross_validation.train_test_split(temp, trainset.train_labels,test_size=0.99)
#X_test = X_test[:5000]
#y_test = y_test[:5000]
#
##import matplotlib.pyplot as plt
#print(X_train.shape,type(X_train))
##image_sample = X_train[0].reshape(32,32,3);
##plt.figure();
##plt.imshow(image_sample)
##print(y_train[0])
##plt.show()
#
#temp = [];
#print("TWO")
#for i in testset.test_data:  #convert to suitable form
#    temp.append(i.reshape(1,-1)[0])
#TESTX = np.asarray(temp)
#TESTY = testset.test_labels;
#randomErr = [];
#print('X train :',X_train.shape)
#print('X test  :',X_test.shape)
#print("TEST    :",TESTX.shape);
#
##TEMP TEST:
#print("THREE"+str(X_train.shape))
##Random training:
#Rerror = []
#for i in range(1,8):
#    print("HELLO")
#    clf.fit(X_train,y_train);
#    pred = clf.predict(TESTX[:1000])
#    Rerror.append(metrics.f1_score(pred,TESTY[:1000], average='macro'))
#    print(i,Rerror[-1])
#    X_train = np.append(X_train,X_test[(i-1)*100:i*100],axis=0)
#    y_train = np.append(y_train,y_test[(i-1)*100:i*100],axis=0)
#
#pred = clf.predict(TESTX[:1000])
#err  = metrics.f1_score(pred, TESTY[:1000] ,average='macro')
#print("DONE with Random");
#print("Final test error : "+str(err));
#
#Terror = [];
#print("Begin Smart:");
#print("Training.....");
#X_train,y_train = X_train[:500],np.asarray(y_train[:500])
#y_test = np.asarray(y_test);
#idx=np.asarray([]).astype('int');
#for i in range(1,8):
#    print("HELLO2")
#    clf.fit(X_train,y_train);
#    idx = np.append(idx,ClosestToLine(clf,X_test,100,idx,t))
#    pred = clf.predict(TESTX[:1000])
#    Terror.append(metrics.f1_score(pred,TESTY[:1000], average='macro'))
#    print(i,Terror[-1])
#    print(idx.shape)
#    X_train = np.append(X_train,X_test[idx[-100:]],axis=0)
#    y_train = np.append(y_train,y_test[idx[-100:]],axis=0)
#
#print("DONE");
#pred = clf.predict(TESTX[:1000])
#err  = metrics.f1_score(pred, TESTY[:1000] ,average='macro')
#print(err);
#Rerror = np.diff(np.asarray(Rerror));
#Terror = np.diff(np.asarray(Terror));
#import matplotlib.pyplot as plt
#plt.plot(Rerror,'r+',label='Random')
#plt.plot(Terror,'go',label='Smart')
#plt.legend()
#plt.show();
