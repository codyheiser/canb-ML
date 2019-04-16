# -*- coding: utf-8 -*-
'''
CANB8347 Machine Learning Project
Apr2019

Utility functions for testing ML algorithms
'''
# basic matrix/dataframe manipulation
import numpy as np
import pandas as pd
from scipy import interp

# sklearn tools
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import roc_curve, auc, confusion_matrix

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
        conf_matrix = confusion_matrix(splits['test']['labels'][split], prediction)
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


def cm_metrics(cm, pretty_print=False):
    '''calculate common metrics based on confusion matrix (e.g. accuracy, precision, sensitivity, specificity)'''
    assert cm.shape == (2,2), "Confusion matrix must be 2 x 2."

    acc = cm.diagonal().sum()/cm.sum()
    prec = cm[1,1]/cm[:,1].sum()
    sens = cm[1,1]/cm[1,:].sum()
    spec = cm[0,0]/cm[0,:].sum()

    if pretty_print:
        print('Accuracy: {}\nPrecision: {}\nSensitivity: {}\nSpecificity: {}'.format(acc,prec,sens,spec))

    return acc, prec, sens, spec


def roc_kfold(clf, X, y, k):
    '''Run classifier with cross-validation and plot ROC curves'''
    cv = StratifiedKFold(n_splits=k)

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(7,7))
    i = 0
    for train, test in cv.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        i += 1

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Chance', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    plt.close()
