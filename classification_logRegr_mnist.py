#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 14:58:46 2019

@author: dell
"""

from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

banking=pd.read_csv('/home/dell/Documents/datasets/banking.csv')
print(banking)

#mnist=pd.read_csv('/home/dell/Documents/datasets/mnist-original.mat')
#print(mnist)
#print(banking.shape)
#banking.info()
#banking.columns
#call data like BPO like for taking loan per day calls and records
#loan approved or not
#banking.education

banking=banking.dropna()
print(banking.education.unique())
#banking.eduaction.unique()
#np.ehere takes 3 arg condn,name or value to replace,column name
banking['education']=np.where(banking['education']=='basic.9y','Basic',banking['education'])
banking['education']=np.where(banking['education']=='basic.6y','Basic',banking['education'])
banking['education']=np.where(banking['education']=='basic.4y','Basic',banking['education'])
'''
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
#banking=banking.columns.values.tolist()
#modify the category
#in logistic regression we reuire numeric values not categorical
for i in cat_vars:
    cat_list='var'+'_'+i
    cat_list=pd.get_dummies(banking[i],prefix=i)
    banking1=banking.join(cat_list)
    banking=banking1
print(banking)

#take all the columns
cols=[""]
#x=banking.drop('y',1)
#y=banking['y']
#y=np.float(y)
#x=np.float(x)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
lr_accuracy=logreg.score(x_test,y_test)
print('Accuracy of logistic regression:%f'%lr_accuracy)
'''