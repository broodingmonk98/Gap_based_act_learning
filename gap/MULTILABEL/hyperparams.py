from sklearn.datasets import fetch_mldata
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.multiclass import OneVsRestClassifier as ovr
from sklearn.model_selection import train_test_split
import numpy as np

random_state=1
test_size=0.4
pool_size= 0.5
iris = datasets.load_iris()
yeast = fetch_mldata('scene-classification')
X = yeast['data'];
y = yeast['target'].transpose().toarray().astype('int32');
c_range = 10.0 ** np.arange(-3, 2)
best = 0;
besti = 0;
bestj = 0;
X_train, X1, y_train, y1 = train_test_split(X,y,random_state=random_state,test_size=test_size)
X_test, X_pool, y_test, y_pool = train_test_split(X1,y1, random_state=random_state,test_size=pool_size)
for i in c_range:
    #for j in dual_range:
        print(i)
        clf = svm.LinearSVC(C=i,dual=False)
        clf = ovr(clf);
        clf.fit(X_train,y_train)
        temp = clf.score(X_test,y_test)
        print(temp)
        if temp>best:
            best = temp;
            besti = i;
            #bestj = j;
clf = ovr(svm.LinearSVC(C=besti,dual=False))
clf.fit(X_train,y_train)
print('Best Score :',best,'best param :',besti)#,bestj)
print('True Score :',clf.score(X_pool,y_pool))
c_range = 10.0 ** np.arange(0, 4)
gamma_range = 10.0 ** np.arange(-2, 2);

for i in c_range:
    for j in gamma_range:
        print(i,j)
        clf = svm.SVC(kernel='rbf',C=i,gamma=j)
        clf = ovr(clf);
        clf.fit(X_train,y_train)
        temp = clf.score(X_test,y_test)
        print(temp)
        if temp>best:
            best = temp;
            besti = i;
            bestj = j;

clf = ovr(svm.SVC(kernel='rbf',gamma=bestj,C=besti))
clf.fit(X_train,y_train)
print('Best Score :',best,'best param :',besti,bestj)
print('True Score :',clf.score(X_pool,y_pool))

#parameters = dict(estimator__clf__gamma=gamma_range,estimator__clf__c=c_range)

#svc = ovr(svm.LinearSVC())
#clf = GridSearchCV(svc, parameters)
#clf.fit(X,y))
