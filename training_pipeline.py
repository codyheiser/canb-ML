# -*- coding: utf-8 -*-
'''
CANB8347 Machine Learning Project
Apr2019

Classifier training pipeline for 'vlbw' dataset
'''
from ml_utils import *
from sklearn.ensemble import RandomForestClassifier
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build classifier for live birth data and predict labels. CANB8347 machine learning project.')
    parser.add_argument('train', type=str, help='path to preprocessed training set as .csv')
    parser.add_argument('pred', type=str, help='path to preprocessed prediction set as .csv')
    parser.add_argument('-o','--outloc', type=str, default='out.csv', help='path to output .csv file')
    args = parser.parse_args()

    train = pd.read_csv(args.train) # read in training data
    train_labels = train['dead'] # pull out 'dead' as labels to train on and predict
    train.drop('dead', axis=1, inplace=True) # remove labels from training features
    train_norm = normalize(train, axis=1, norm='l1') # normalize each column to fractional representation between 0 and 1

    # initiate random forest using parameters optimized on training data using RandomizedSearchCV
    # see training_presentation.ipynb for detailed workflow and results
    rf = RandomForestClassifier(n_estimators=700, max_depth=100, min_samples_split=2,
    min_samples_leaf=4, max_features='auto', bootstrap=False, random_state=0)

    rf.fit(X=train_norm, y=train_labels) # fit rf to training data


    pred = pd.read_csv(args.pred) # read in data to predict on
    pred_labels = pred['dead'] # pull out 'dead' as labels to try to predict
    pred.drop('dead', axis=1, inplace=True)

    # ensure all columns are represented in testing data, and in the same order
    for clmn in train.columns:
        if clmn not in pred.columns:
            print('Adding column "{}" to prediction set'.format(clmn))
            pred[clmn] = 0

    pred = pred[train.columns] # make sure columns are in the same order for when you classify on matrix

    pred_norm = normalize(pred, axis=1, norm='l1') # normalize each column to fractional representation between 0 and 1
    prediction = rf.predict(X=pred_norm)
    pred['dead_predict'] = prediction # predict labels in prediction set and put back in pred df
    pred['dead'] = pred_labels # put truth labels back into dataframe for export
    pred.to_csv(args.outloc, index=False) # write resulting df to .csv file

    conf_matrix = confusion_matrix(prediction, pred_labels)
    cm_metrics(conf_matrix, pretty_print=True)
    plot_cm(conf_matrix)
    print('\nDone!')
