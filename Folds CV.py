
# coding: utf-8

# In[5]:

import pandas,sklearn
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import *
import random
import numpy as np
from sklearn.feature_selection import SelectFromModel, chi2, f_classif, f_oneway
import matplotlib.pyplot as plt
from sklearn import datasets, svm
from sklearn.feature_selection import SelectPercentile

train1 = pandas.read_csv("D:\\Downloads\\train.csv")
train1 = train1.iloc[np.random.permutation(len(train1))]

def feats():
    f = "feat_"
    ff = []
    for i in range(1,94):
        ff.append(f+str(i))
    return ff

predictors = feats()

xtrain = train1[predictors][:2000]
ytrain = train1["target"][:2000]


# In[7]:

scores = []
clf = svm.SVC(kernel='linear')
for folds in range(10,30,5): 
#     selector = sklearn.feature_selection.SelectKBest(chi2,k=i) 
#     selector.fit(xtrain, ytrain)
    score = cross_validation.cross_val_score(clf, xtrain, ytrain, cv=folds)
    scores.append(score.mean()*100)
    
print scores


# In[ ]:



