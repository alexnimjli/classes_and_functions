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

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd

# +
import sys
sys.path
sys.path.insert(0, '../classes_functions')

from quick_model_tests import quick_model_tests

# +
seed = 44
np.random.seed(seed)

df_train = pd.read_csv('../data/processed_data/df_train_split_ppc.csv')
df_test = pd.read_csv('../data/processed_data/df_test_split_ppc.csv')

y_train = df_train['isFraud']
X_train = df_train.drop('isFraud', axis = 1)
y_test = df_test['isFraud']
X_test = df_test.drop('isFraud', axis = 1)

# +
GaussianNB = GaussianNB()
SGDClassifier = SGDClassifier()
RandomForest = RandomForestClassifier(n_estimators=10)
XGBClassifier = XGBClassifier()

scores_df = pd.DataFrame(columns = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'fit_time'])

models = [GaussianNB, SGDClassifier, RandomForest, XGBClassifier]
names = ["Naive Bayes", "SGD Classifier", 'Random Forest Classifier', 'XGB Classifier']
# -

scores_df = quick_model_tests(models, names, X_train, y_train)

scores_df


