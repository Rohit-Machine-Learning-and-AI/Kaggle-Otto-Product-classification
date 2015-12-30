
# coding: utf-8

# In[10]:

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
from scipy.stats import ttest_ind

train1 = pandas.read_csv("D:\\Downloads\\train.csv")
train1 = train1.iloc[np.random.permutation(len(train1))]

def feats(remove):
    f = "feat_"
    ff = []
    f_desc = [34, 11, 25, 60, 14, 3, 27, 62, 46, 40, 36, 67, 26, 69, 54, 61, 39, 8, 80, 90, 88, 28, 50, 75, 42, 15, 24, 76, 20, 9, 86, 38, 4, 59, 58, 57, 17, 82, 2, 43, 68, 72, 41, 53, 64, 45, 49, 18, 23, 32, 70, 79, 35, 19, 33, 85, 92, 7, 22, 66, 83, 71, 13, 29, 37, 91, 55, 73, 78, 48, 47, 56, 21, 30, 44, 81, 87, 1, 10, 31, 52, 77, 16, 5, 89, 84, 74, 93, 63, 65, 51, 12, 6]
    for i in range(len(f_desc) - remove):
        ff.append(f+str(f_desc[i]))
    #print ff
    return ff

folds = 20 #selected after analysis
no_of_features = 76 #selected after analysis

predictors = feats(93 - no_of_features)

for i in predictors:
    train1[i] = train1[i].fillna(train1[i].median())

xtrain = train1[predictors][:500]
ytrain = train1["target"][:500]

clf = svm.SVC(C=100, gamma=0.0001, kernel='rbf', probability=True)
clf.fit(xtrain, ytrain)
predictions = clf.predict_proba(train1[predictors])
print predictions.shape


def save_csv(filename, data):
    header = "id,Class_1,Class_2,Class_3,Class_4,Class_5,Class_6,Class_7,Class_8,Class_9"
    ids = np.array( [ [i+1] for i in range(data.shape[0]) ] )
    data = np.concatenate( (ids,data), axis=1)
    fmt = "%.10f"
    np.savetxt(filename, data, delimiter=",", header=header, comments="",
               fmt=[ "%.0f", fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt, fmt] )


# In[9]:

save_csv("D:\\Downloads\\myfile.csv", predictions)


# In[ ]:



