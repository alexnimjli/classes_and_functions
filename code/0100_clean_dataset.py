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

# +
import pandas as pd
import numpy as np

import sys
sys.path

sys.path.insert(0, '../classes_functions')

from data_processing_class import DataProcessing

# +
# Train-test-split
df_tran = pd.read_csv("../data/original_data/train_transaction.csv", index_col = 'TransactionID')
df_id = pd.read_csv("../data/original_data/train_identity.csv", index_col = 'TransactionID')

df_tot = df_tran.merge(df_id, how = 'left', left_on='TransactionID', right_on='TransactionID')
df_train_split = pd.read_csv("../data/processed_data/df_train_split.csv")
df_test_split = pd.read_csv("../data/processed_data/df_test_split.csv")
df_train_split_pp = DataProcessing(df_train_split, 'isFraud')
df_train_split_pp.threshold_col_del(0.25)

df_train_split_pp.extract_timestamps()

numerical_cols = []
categorical_cols = []

for col in df_train_split_pp.X.columns:
    if df_train_split_pp.X[col].dtype != 'object':
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)
df_train_split_pp.lblencoder()
df_train_split_pp.fill_null(categorical_cols, 'mode')
df_train_split_pp.fill_null(numerical_cols, 'median')

df_train_split_pp.balancesample("over")
df_train_split_pp.standardiser()
df_train_split_pp.pca_reduction(0.95)
df_train_split_pp.X.head()
df_train_split_ppc = pd.concat([df_train_split_pp.X, df_train_split_pp.y], axis=1, sort=False)
df_train_split_ppc.to_csv("../data/processed_data/df_train_split_ppc.csv", index=False)

df_test_split_pp = DataProcessing(df_test_split, 'isFraud')
df_test_split_pp.threshold_col_del(0.25)
df_test_split_pp.extract_timestamps()

numerical_cols = []
categorical_cols = []

for col in df_test_split_pp.X.columns:
    if df_train_split_pp.X[col].dtype != 'object':
        numerical_cols.append(col)
    else:
        categorical_cols.append(col)
df_test_split_pp.lblencoder()
df_test_split_pp.fill_null(categorical_cols, 'mode')
df_test_split_pp.fill_null(numerical_cols, 'median')
df_test_split_pp.standardiser()

df_test_split_pp.X.head()
df_test_split_ppc = pd.concat([df_test_split_pp.X, df_test_split_pp.y], axis=1, sort=False)
df_test_split_ppc.to_csv("../data/processed_data/df_test_split_ppc.csv", index=False)

# -


