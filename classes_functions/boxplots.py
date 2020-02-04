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
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def simple_box_plot(df, x_attrib, y_attrib):
    f, axes = plt.subplots(ncols=1, figsize=(7,7))

    sns.boxplot(x=y_attrib, y=x_attrib, data=df)
    axes.set_title(x_attrib)

    
    for i in df[y_attrib].unique():
        print("Median for '{}'': {}".format(i, df[x_attrib][df[y_attrib] == i].median()))

    plt.show()


def remove_outliers(df, x_attrib, y_attrib):

    for i in df[y_attrib].unique():
        
        m, n = df.shape
        #print('Number of rows: {}'.format(m))
        
        remove_list = df[x_attrib].loc[df[y_attrib] == i].values
        q25, q75 = np.percentile(remove_list, 25), np.percentile(remove_list, 75)
        #print('Lower Quartile: {} | Upper Quartile: {}'.format(q25, q75))
        iqr = q75 - q25
        #print('iqr: {}'.format(iqr))

        cut_off = iqr * 1.5
        lower, upper = q25 - cut_off, q75 + cut_off
        #print('Cut Off: {}'.format(cut_off))
        #print('Lower Extreme: {}'.format(lower))
        #print('Upper Extreme: {}'.format(upper))

        outliers = [x for x in remove_list if x < lower or x > upper]
        #print('Number of Outliers for {} Cases: {}'.format(i, len(outliers)))
        #print('outliers:{}'.format(outliers))

        for d in outliers:
            #delete_row = new_df[new_df[y_attrib]==i].index
            #new_df = new_df.drop(delete_row)
            df = df[df[x_attrib] != d]
        
        m, n = df.shape
        #print('Number of rows for new dataframe: {}\n'.format(m))
    
    new_df = df
    
    #print('----' * 27)
    return new_df


