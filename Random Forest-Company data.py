# -*- coding: utf-8 -*-
"""
Created on Fri May  1 12:03:52 2020

@author: patel
"""

import pandas as pd
import numpy as np

data=pd.read_csv("C:\\Users\\patel\\Downloads\\Company_Data (1).csv")
#data with out "High"
data2=pd.read_csv("C:\\Users\\patel\\Downloads\\Company_Data (1).csv")
data2.dtypes 

num=data['Sales']
data['High']=np.where(num<=9,'0','1') ##0='no',1='yes'

obj_data2=data2.select_dtypes(include=('int64','float64','object'))
cleanup_nums2={"ShelveLoc":{"Good":1,"Bad":0,"Medium":2} }
obj_data2.replace(cleanup_nums2,inplace=True)
data2=obj_data2
#obj_data2['ShelveLoc']=obj_data2['ShelveLoc'].astype('int64')


num2=data2['US']
data2['US']=np.where(num2=='YES',1,0)

num3=data2['Urban']
data2['Urban']=np.where(num3=='YES',1,0)

from sklearn.preprocessing import StandardScaler
norm=StandardScaler()
norm.fit(data2)
norm_data=norm.transform(data2)
type(norm_data)

np.savetxt("norm_data.csv", norm_data,fmt="%d", delimiter=",")

data2=pd.read_csv("norm_data.csv")

data2.shape

#train and test

from sklearn.model_selection import train_test_split
x=data2.iloc[:399,1:11]
x.shape

y=data2.iloc[:399,0]
y.shape

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

y_test




data2.dtypes

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

# Accuracy = train
from sklearn import metrics
print("Accuracy:",metrics.accuracy_score(y_test, preds))


# KFold validation

from sklearn.model_selection import KFold

X=data2.iloc[:399,1:11].to_numpy()

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
print("Accuracy:",metrics.accuracy_score(y_test, preds))  #98.4%
