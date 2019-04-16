# -*- coding: utf-8 -*-
'''
CANB8347 Machine Learning Project
Apr2019

Preprocessing and imputation pipeline for 'vlbw' dataset
'''
from ml_utils import *
from sklearn.ensemble import RandomForestClassifier
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process and build classifier for live birth data. CANB8347 machine learning project.')
    parser.add_argument('file', type=str, help='path to input dataframe as .csv')
    parser.add_argument('-o','--outloc', type=str, default='out.csv', help='path to output .csv file')
    args = parser.parse_args()

    # read in raw training data
    vlbw = pd.read_csv(args.file)

    # PREPROCESSING #

    # get rid of index axis
    vlbw.drop('Unnamed: 0', axis=1, inplace=True)

    # drop rows with NaN in column 'twn'; these have a lot of other missing info
    vlbw.dropna(subset=['twn'], inplace=True)
    # drop times that we determined to be irrelevant
    vlbw.drop(['birth','exit','hospstay','year'], axis=1, inplace=True)

    # add labor length of 0 for any abdominal births without any value already assigned
    vlbw.loc[(vlbw.delivery=='abdominal') & (vlbw.lol.isnull()), 'lol'] = 0

    # impute zero for the missing values in three numeric columns
    vlbw.loc[vlbw.magsulf.isnull(), 'magsulf'] = 0
    vlbw.loc[vlbw.meth.isnull(), 'meth'] = 0
    vlbw.loc[vlbw.toc.isnull(), 'toc'] = 0

    # replace categories with numeric levels based on confidence of diagnosis
    for col in ['pvh','ivh','ipe']:
        vlbw.loc[vlbw[col]=='absent', col] = 0
        vlbw.loc[vlbw[col]=='possible', col] = 1
        vlbw.loc[vlbw[col]=='definite', col] = 2
        vlbw.loc[:,col] = vlbw[col].astype('float64') # ensure numeric datatype

    # perform numerization on remaining string columns
    for feature, datatype in zip(vlbw.dtypes.index, vlbw.dtypes):
        if datatype == 'object':
            vlbw = numerize(vlbw, feature)

    # there's only one observation where we don't know whether or not they were born at Duke.  Let's call it a Dukie.
    vlbw.loc[(vlbw['inout_born at Duke']==0)&(vlbw.inout_transported==0), 'inout_born at Duke'] = 1
    vlbw.drop('inout_nan', axis=1, inplace=True)

    # drop unknown race, sex, and delivery bc there's not that many missing values
    vlbw = vlbw[(vlbw.race_nan!=1) & (vlbw.sex_nan!=1) & (vlbw.delivery_nan!=1)]
    vlbw.drop(['race_nan','delivery_nan','sex_nan'], axis=1, inplace=True)

    # IMPUTATION #

    # insert median lol for vaginal births into missing values
    lol_val = np.nanmedian(vlbw[vlbw.delivery_vaginal==1]['lol'])
    vlbw.loc[vlbw.lol.isnull(), 'lol'] = lol_val

    # replace missing values with the median of the other values
    vlbw.fillna(value={'lowph':np.nanmedian(vlbw.lowph),'pltct':np.nanmedian(vlbw.pltct),'gest':np.nanmedian(vlbw.gest),
    'apg1':np.nanmedian(vlbw.apg1),'vent':np.nanmedian(vlbw.vent),'pneumo':np.nanmedian(vlbw.pneumo),'pda':np.nanmedian(vlbw.pda),
    'cld':np.nanmedian(vlbw.cld)}, inplace=True)

    # impute 'pvh', 'ivh', and 'ipe' using a random forest classifier
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)

    pvh_prediction = impute_with_classifier(df=vlbw.drop(['ivh','ipe'], axis=1), col='pvh', clf=rf)
    ivh_prediction = impute_with_classifier(df=vlbw.drop(['pvh','ipe'], axis=1), col='ivh', clf=rf)
    ipe_prediction = impute_with_classifier(df=vlbw.drop(['ivh','pvh'], axis=1), col='ipe', clf=rf)

    vlbw.loc[vlbw.pvh.isnull(), 'pvh'] = pvh_prediction
    vlbw.loc[vlbw.ivh.isnull(), 'ivh'] = ivh_prediction
    vlbw.loc[vlbw.ipe.isnull(), 'ipe'] = ipe_prediction

    # write results to file
    vlbw.to_csv(args.outloc, index=False)
    print('\nDone!')
