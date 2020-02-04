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
import seaborn as sns
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def corr_matrix(df, attribute_list, key_attribute):
    new_df = pd.DataFrame()
    for i in attribute_list:
        new_df[i] = df[i]
            
    matrix = new_df.corr()
    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(matrix, vmax=.8, square=True, cmap="YlGnBu")
    
    print(matrix[key_attribute].sort_values(ascending=False))
    
    return 
