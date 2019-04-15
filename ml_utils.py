# -*- coding: utf-8 -*-
'''
CANB8347 Machine Learning Project
Apr2019

Utility functions for testing ML algorithms
'''
# basic matrix/dataframe manipulation
import numpy as np
import pandas as pd

# sklearn tools
from sklearn.model_selection import KFold
from sklearn import metrics

# plotting tools
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style = 'white')


def numerize(df, col, drop=True):
    '''
    make categorical data numeric from 0 - n categories
        df = dataframe
        col = column to numerize into n_categories columns
        drop = drop original column or retain in df?
    '''
    temp = df.copy(deep=True) # copy df so you don't affect it

    for cat in temp[col][temp[col].notnull()].unique():
        # for each categorical value, create a new column with binary values for T/F
        temp[col+'_'+cat] = (temp[col]==cat)*1

    if drop:
        return temp.drop(col, axis=1)

    else:
        return temp


def kfold_split(data, labels, n_splits, seed=None, shuffle=True):
        '''
        split obs using k-fold strategy to cross-validate
            returns: dictionary with keys ['train','test'], which each contain a dictionary with keys ['data','labels'].
                values for ['data','labels'] are list of matrices/vectors
            ex: train data for the 3rd split can be indexed by `split['train']['data'][2]`,
                and its corresponding labels by `split['train']['labels'][2]`
        '''
        kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed) # generate KFold object for splitting data
        splits = {'train':{'data':[],'labels':[]}, 'test':{'data':[],'labels':[]}} # initiate empty dictionary to dump matrix subsets into

        for train_i, test_i in kf.split(data):
            splits['train']['data'].append(data[train_i,:])
            splits['train']['labels'].append(labels[train_i])
            splits['test']['data'].append(data[test_i,:])
            splits['test']['labels'].append(labels[test_i])

        return splits


def validator(splits, classifier):
    '''loops through kfold_split object and calculates confusion matrix and accuracy scores for given classifier'''
    for split in range(0, len(splits['train']['data'])):
        classifier.fit(splits['train']['data'][split], splits['train']['labels'][split])
        prediction = classifier.predict(splits['test']['data'][split])
        conf_matrix = metrics.confusion_matrix(splits['test']['labels'][split], prediction)
        score = classifier.score(splits['test']['data'][split], splits['test']['labels'][split])

        print('\nSplit {}: {}\n{}'.format(split,score,conf_matrix))


def plot_cm(cm):
    '''plot confusion matrix using seaborn for pretty output'''
    plt.figure(figsize=(3,3))
    sns.heatmap(cm, annot=True, linewidths=.5, square = True, cmap = 'Blues_r', cbar=False, annot_kws={'fontsize':18})
    plt.ylabel('Actual Label', fontsize=14)
    plt.xlabel('Predicted label', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    score = cm.diagonal().sum()/cm.sum()
    plt.title('Accuracy: {0}'.format(np.round(score,3)), size = 14)
    plt.show()
    plt.close()


def cm_metrics(cm):
    '''calculate common metrics based on confusion matrix (e.g. accuracy, precision, sensitivity, specificity)'''
    assert cm.shape == (2,2), "Confusion matrix must be 2 x 2."
    acc = cm.diagonal().sum()/cm.sum()
    prec = cm[1,1]/cm[:,1].sum()
    sens = cm[1,1]/cm[1,:].sum()
    spec = cm[0,0]/cm[0,:].sum()
    return acc, prec, sens, spec
