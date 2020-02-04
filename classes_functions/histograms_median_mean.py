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

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# +
dodger_blue = '#1E90FF'
crimson = '#DC143C'
lime_green = '#32CD32'
red_wine = '#722f37'
white_wine = '#dbdd46' 

def plot_histograms(df, x_attribute, n_bins, x_max, y_attribute):
    
    #this removes the rows with nan values for this attribute  
    df = df.dropna(subset=[x_attribute]) 
    
    print ("Mean: {:0.2f}".format(df[x_attribute].mean()))
    print ("Median: {:0.2f}".format(df[x_attribute].median()))
           
    df[x_attribute].hist(bins= n_bins, color= crimson)
    
    #this plots the mean and median 
    plt.plot([df[x_attribute].mean(), df[x_attribute].mean()], [0, 60000],
        color='black', linestyle='-', linewidth=2, label='mean')
    plt.plot([df[x_attribute].median(), df[x_attribute].median()], [0, 60000],
        color='black', linestyle='--', linewidth=2, label='median')
    
    plt.xlim(xmin=0, xmax = x_max)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.title(x_attribute)
    plt.legend(loc='best')
    plt.show()

    df[df[ y_attribute]==0][x_attribute].hist(bins=n_bins, color = crimson, label='No default')

    print ("Y Mean: {:0.2f}".format(df[df[y_attribute]==0][x_attribute].mean()))
    print ("Y Median: {:0.2f}".format(df[df[ y_attribute]==0][x_attribute].median()))
    
    plt.plot([df[df[ y_attribute]==0][x_attribute].mean(), df[df[ y_attribute]==0][x_attribute].mean()], 
            [0, 60000], color='r', linestyle='-', linewidth=2, label='Y mean') 
    plt.plot([df[df[ y_attribute]==1][x_attribute].mean(), df[df[ y_attribute]==1][x_attribute].mean()], 
            [0, 60000], color='b', linestyle='-', linewidth=2, label='N mean')
 
    df[df[ y_attribute]==1][x_attribute].hist(bins=n_bins, color = lime_green, label='Default')
    
    print ("N Mean: {:0.2f}".format(df[df[ y_attribute]==1][x_attribute].mean()))
    print ("N Median: {:0.2f}".format(df[df[ y_attribute]==1][x_attribute].median()))
    
    plt.plot([df[df[ y_attribute]==0][x_attribute].median(), df[df[ y_attribute]==0][x_attribute].median()], 
            [0, 60000], color='r', linestyle='--', linewidth=2, label='Y median') 
    plt.plot([df[df[ y_attribute]==1][x_attribute].median(), df[df[ y_attribute]==1][x_attribute].median()], 
            [0, 60000], color='b', linestyle='--', linewidth=2, label='N median')
    
    plt.xlim(xmin=0, xmax = x_max)
    
    plt.title(x_attribute)
    plt.xlabel(x_attribute)
    plt.ylabel('COUNT')
    plt.legend(loc='best')
    plt.show()    
    return
    


# -


