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

from confusion_matrices import confusion_matrices
# -

df_id = pd.read_csv("../data/original_data/train_identity.csv", index_col = 'TransactionID')

y = df_id['id_36']
F = pd.get_dummies(y)['F']
T = pd.get_dummies(y)['T']

confusion_matrices(F,F)


