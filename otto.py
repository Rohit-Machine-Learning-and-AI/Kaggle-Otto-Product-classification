# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
import pandas
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
import random
import os
from scipy.stats import ttest_ind

if __name__ == '__main__':
    def select_n_rows_per_class(n):
        classes = []
        classes.append(random.randint(10,1779-n))
        classes.append(random.randint(1939,17901-n))
        classes.append(random.randint(18061,25905-n))
        classes.append(random.randint(26065,28596-n))
        classes.append(random.randint(28756,31335-n))
        classes.append(random.randint(31495,45470-n))
        classes.append(random.randint(45630,48309-n))
        classes.append(random.randint(48469,56773-n))
        classes.append(random.randint(56933,61728-n))

        train = pandas.concat([train1[1:2]])
        for i in classes:
            train = pandas.concat([train, train1[i:i+n]])
        #print("Subset: {}".format(classes))
        return train
        
    def feats(remove):
        f = "feat_"
        ff = []
        f_desc = [34, 11, 25, 60, 14, 3, 27, 62, 46, 40, 36, 67, 26, 69, 54, 61, 39, 8, 80, 90, 88, 28, 50, 75, 42, 15, 24, 76, 20, 9, 86, 38, 4, 59, 58, 57, 17, 82, 2, 43, 68, 72, 41, 53, 64, 45, 49, 18, 23, 32, 70, 79, 35, 19, 33, 85, 92, 7, 22, 66, 83, 71, 13, 29, 37, 91, 55, 73, 78, 48, 47, 56, 21, 30, 44, 81, 87, 1, 10, 31, 52, 77, 16, 5, 89, 84, 74, 93, 63, 65, 51, 12, 6]
        for i in range(len(f_desc) - remove):
            ff.append(f + str(f_desc[i]))
        return ff

#singlerun:
    def singlerun_logistic(folds):
        scores = cross_validation.cross_val_score(LogisticRegression(random_state=1), train[predictors], train["target"], cv=folds)
        return scores.mean()*100
        print("Logistic Regression: {} %".format(scores.mean()*100))

    def singlerun_randomforests(folds, estims, samples_per_split, samples_per_leaf):
        r = RandomForestClassifier(random_state=1, n_estimators=estims, min_samples_split=samples_per_split, min_samples_leaf=samples_per_leaf)
        scores = cross_validation.cross_val_score(r, train[predictors], train["target"], cv=folds)
        return scores.mean()*100
        print("Random Forests: {} %".format(scores.mean()*100))
        
    train1 = pandas.read_csv("train.csv")

    train = train1 #entire dataset
    folds = 20 #selected after analysis
    no_of_features = 76 #selected after analysis
    predictors = feats(93 - no_of_features)
    
  #Fill missing data, if any
    for i in predictors:
        train[i] = train[i].fillna(train[i].median())

    lr, rf = [], []
    for i in range(1, 2):
        #train = select_n_rows_per_class(100) #selecting mentioned no of rows for each class label
        lr.append(singlerun_logistic(20))
        rf.append(singlerun_randomforests(20, 200, 6, 1))
      
    lrm = sum(lr)/len(lr)
    rfm = sum(rf)/len(rf)
    print("\nLogistic Accuracy mean: {}\nRandomForests Accuracy mean: {}".format(lrm, rfm))
    #print("\n{}\n{}".format(lrm, rfm))
    #print("\n{}\n{}".format(lr, rf))
    
   #Welchâ€™s t-test --> does not assume equal population variance
    statistic, pvalue = ttest_ind(lr, rf, equal_var=False)
    print "\nWelchâ€™s  t-test: statistic = %.5f and pvalue = %.5f" % (statistic, pvalue)
    #print "\n%.5f\n%.5f" % (statistic, pvalue)
    statistic, pvalue = ttest_ind(lr, rf, equal_var=True)
    print "\n%.5f\n%.5f" % (statistic, pvalue)
    #print "Standard t-test: statistic = %.5f and pvalue = %.10f" % (statistic, pvalue)

#referred kaggle titanic tutorial
    
    
  #Feature selection
    selector = SelectKBest(f_classif, k=76)
    selector.fit(train[predictors], train["target"])
    scores = -np.log10(selector.pvalues_)
    plt.bar(range(len(predictors)), scores)
    plt.xticks(range(len(predictors)), predictors, rotation='vertical')
    plt.show()

