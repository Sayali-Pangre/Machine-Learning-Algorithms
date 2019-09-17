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
from sklearn.feature_selection import RFE
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

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
#banking=banking.columns.values.tolist()
#modify the category
#in logistic regression we reuire numeric values not categorical
for i in cat_vars:
    cat_list='var'+'_'+i    #new column names
    cat_list=pd.get_dummies(banking[i],prefix=i)    #
    banking1=banking.join(cat_list)
    banking=banking1
print(banking)
#banking.columns

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
banking_vars=banking.columns.values.tolist()
#print(banking_vars)

to_keep=[i for i in banking_vars if i not in cat_vars]   #remove the previous col bcz by getdummis already filterd out columns
banking_final=banking[to_keep]
banking_final.columns.values

banking_final_vars=banking_final.columns.values.tolist()
#print(banking_final_vars)
y=['y']
x=[i for i in banking_final_vars if i not in y]

#print x and y
#main algo
logreg=LogisticRegression()
rfe=RFE(logreg,18)
rfe=rfe.fit(x_data,y_data)
print(rfe.support_)
print(rfe.ranking_)

cols=["previous","euribor3m","job_blue-collar","job_retired","job_services","job_student","default_no","month_aug","month_dec","month_jul","month_nov","month_jul","month_oct","month_sep","day_of_week_fri","day_of_week_wed","poutcome_failure","poutcome_nonexistent","poutcome_success"]
y_data=banking_final[y]
x_data=banking_final[cols]
#take all the columns
x_train,x_test,y_train,y_test=train_test_split(x_data,y_data,test_size=0.2,random_state=0)

logreg=LogisticRegression()
logreg.fit(x_train,y_train)

y_pred=logreg.predict(x_test)
lr_accuracy=logreg.score(x_test,y_test)

from sklearn.metrics import confusion_matrix
confusion_matrix=confusion_matrix(y_test,y_pred)
print(confusion_matrix)

from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc=roc_auc_score(y_test,logreg.predict(x_test))
fpr,tpr,thresholds=roc_curve(y_test,logreg.predict_proba(x_test)[:,1])
plt.figure()
plt.plot(fpr,tpr,label='Logistic Regression(area=%0.2f)' %logit_roc_auc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0.0,1.05])
plt.ylim([0.0,1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True positive rate')
plt.title('Receiver operating characteristic')
plt.legend(loc=4)
#plt.savefig('Log_ROC')
plt.show()
print(thresholds)
print(logit_roc_auc)
print('Accuracy of logistic regression:%f'%lr_accuracy)
#misclassification rate
#if y_pred != y_test:
    

