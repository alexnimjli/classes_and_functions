# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import cross_validate

# +
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier

#GaussianNB = GaussianNB()
#SGDClassifier = SGDClassifier()
#RandomForest = RandomForestClassifier(n_estimators=10)
#XGBClassifier = XGBClassifier()

#models = [GaussianNB, SGDClassifier, RandomForest, XGBClassifier]
#names = ["Naive Bayes", "SGD Classifier", 'Random Forest Classifier', 'XGB Classifier']

# -

def quick_model_tests(models, names, X_train, y_train):
    
    scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time'])
    
    for model, name in zip(models, names):
        temp_list = []
        #print(name)

        model.fit(X_train, y_train)
        scores = cross_validate(model, X_train, y_train,
                                scoring=('accuracy', 'precision', 'recall', 'f1', 'roc_auc'),
                                return_train_score=True, cv=10)

        for score in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
            mean_score = scores['test_'+score].mean()
            #print('{} mean : {}'.format(score, mean_score))
            temp_list.append(mean_score)

        temp_list.append(scores['fit_time'].mean())
        #print('average fit time: {}'.format(scores['fit_time'].mean()))
        #print("\n")
        scores_df.loc[name] = temp_list
        
    return scores_df


