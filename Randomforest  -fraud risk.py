# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 12:54:59 2020

@author: patel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data=pd.read_csv("C:\\Users\\patel\\Downloads\\Fraud_check (3).csv")
#with out "status"
data2=pd.read_csv("C:\\Users\\patel\\Downloads\\Fraud_check (3).csv")
data


num=data['Taxable.Income']
data['Status']=np.where(num<=30000,'risky','good')
data.head()
data.dtypes

data.columns
colnames=list(data.columns)
colnames

obj_data2=data2.select_dtypes(include=('int64','float64','object'))
#obj_data['Status']=obj_data['Status'].astype('int64')

cleanup_nums2={"Undergrad":{"YES":1,"NO":0},"Marital.Status":{"Single":1,"Married":2,"Divorced":3},"Urban":{"YES":1,"NO":0} }
obj_data2.replace(cleanup_nums2,inplace=True)
data2=obj_data2
obj_data2['Urban']=obj_data2['Urban'].astype('int64')
obj_data2['Undergrad']=obj_data2['Undergrad'].astype('int64')
obj_data2['Marital.Status']=obj_data2['Marital.Status'].astype('int64')

data2


obj_data=data.select_dtypes(include=('int64','float64','object'))

cleanup_nums={"Status":{"good":1,"risky":0}}
obj_data.replace(cleanup_nums,inplace=True)
data=obj_data
obj_data['Status']=obj_data['Status'].astype('int64')

data


x=data2
x

y=data.Status
y

#test,train split
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
y_train

x_train.shape,x_test.shape
y_train.shape,y_test.shape


from sklearn.ensemble import  RandomForestClassifier 
#as rc

rc=RandomForestClassifier(criterion="entropy",oob_score=True)
rc.fit(x_train,y_train)

rc.estimators_ # 
rc.classes_ # class labels (output)
rc.n_classes_ # Number of levels in class labels 
rc.n_features_  # Number of input features in model 8 here.

rc.n_outputs_ # Number of outputs when fit performed

rc.oob_score_
rc.predict(x_train)

preds=rc.predict(x_test)
preds

# Accuracy = test
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, preds))

import seaborn as sns
sns.violinplot(x=y_test,y=preds)


# KFold validation

from sklearn.model_selection import KFold

X=data2.to_numpy()


k = KFold(n_splits=2) # Define the split - into 2 folds 
k.get_n_splits(X) # returns the number of splitting iterations in the cross-validator
print(k) 
KFold(n_splits=2, random_state=None, shuffle=False)
k.split(X)

trainK=[]
testK=[]

for train_index, test_index in k.split(X):
    print(train_index , test_index)
    
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
X[train_index].shape  ; y[train_index].shape  

rc=RandomForestClassifier(criterion="entropy",oob_score=True)
rc.fit(X_train,y_train)

rc.estimators_ # 
rc.classes_ # class labels (output)
rc.n_classes_ # Number of levels in class labels 
rc.n_features_  # Number of input features in model 8 here.

rc.n_outputs_ # Number of outputs when fit performed

rc.oob_score_
rc.predict(X_train)

preds=rc.predict(X_test)
preds

# Accuracy = test
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, preds))
