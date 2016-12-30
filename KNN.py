# -*- coding: utf-8 -*-
"""
Created on Fri Dec 30 16:14:46 2016

@author: R16
"""

import numpy as np
def euclid(a,b):
  res=0
  for i in range(len(a)):
    res+=np.power(a[i]-b[i],2)
  return np.sqrt(res)
  
class KNNfromscratch():
  
  def fit(self, X_train, y_train):
    self.X_train = X_train
    self.y_train = y_train
  #prediksi label data test
  def predict(self, X_test, k):
    pred = []
    for row in X_test:
      label = self.closest(row, k)
      pred.append(label)
    return pred
  #menghitung jarak tetangga terdekat untuk setiap row test
  def closest(self, row, k):
    dists=[]
#    best_index=[]
    for i in range(len(self.X_train)):
      dists.append(euclid(row,X_train[i]))
      
    best_index = np.argsort(dists)[:k]
      
    best_label = np.argmax(np.bincount([self.y_train[i] for i in best_index]))
    
    return best_label

from sklearn import  datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target


from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .5)

from sklearn.neighbors import KNeighborsClassifier

classifier = KNeighborsClassifier()
classifier_ = KNNfromscratch()

classifier.fit(X_train,y_train)
classifier_.fit(X_train,y_train)

predictions = classifier.predict(X_test)
predictions_ = classifier_.predict(X_test,3)

from sklearn.metrics import accuracy_score

print("using sklearn KNN")
print(accuracy_score(y_test,predictions))

print("using KNN from scratch")
print(accuracy_score(y_test,predictions_))