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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_bar_graphs(df, attribute, y):
    plt.figure(1)
    plt.subplot(131)
    df[attribute].value_counts(normalize=True).plot.bar(figsize=(22,4),title= attribute)
    
    crosstab = pd.crosstab(df[attribute], df[y])
    crosstab.div(crosstab.sum(1).astype(float), axis=0).plot.bar(stacked=True)
    crosstab.plot.bar(stacked=True)
    
    res = df.groupby([attribute, y]).size().unstack()
    tot_col = 0
    for i in range(len(df[y].unique())):
        tot_col = tot_col + res[res.columns[i]] 
        
    for i in range(len(df[y].unique())):    
        res[i] = (res[res.columns[i]]/tot_col)
        
    #res = df.groupby([attribute, y]).size().unstack()
    #res['percentage of "0"'] = (res[res.columns[1]]/(res[res.columns[0]] + res[res.columns[1]]))
    
    print(res)
    
    return
