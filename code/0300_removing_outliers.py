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
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path

sys.path.insert(0, '../classes_functions')

from boxplots import simple_box_plot, remove_outliers 

# +
seed = 44
np.random.seed(seed)

df_train = pd.read_csv('../data/processed_data/df_train_split_ppc.csv')
df_test = pd.read_csv('../data/processed_data/df_test_split_ppc.csv')

# -

df_train.shape

simple_box_plot(df_train, '0', 'isFraud')

new_df = remove_outliers(df_train, '0', 'isFraud')

test = df_train
for i in df_train.columns:
    if i != 'isFraud':
        test = remove_outliers(test, i, 'isFraud')

simple_box_plot(new_df, '0', 'isFraud')

test.shape


