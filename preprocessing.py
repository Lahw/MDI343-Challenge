"""
Preprocessing script.

@auther Yann Carbonne

You can create the preprocessed data and create a feature selector mask.
With the quick & dirty model, give 0.71556 on public leader board.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import datetime
import calendar
from sklearn.preprocessing import LabelEncoder, Imputer
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.base import BaseEstimator, ClassifierMixin
import matplotlib.pyplot as plt


def get_preproc(with_selector=False):
    """Load preprocessed data with or without the selector mask."""
    train, target, test = np.loadtxt('train.txt'), np.loadtxt('target.txt'), np.loadtxt('test.txt')
    if with_selector:
        selector = np.loadtxt('feature_selector_support.txt')
        selector = np.array(selector, dtype=bool)
        train, test = train[:, selector], test[:, selector]
    return train, target, test


def _create_preprocessing():
    test = pd.read_csv('test.csv', sep=';', na_values='(MISSING)')
    train = pd.read_csv('train.csv', sep=';', na_values='(MISSING)')
    target = train['VARIABLE_CIBLE']
    train.drop('VARIABLE_CIBLE', axis=1, inplace=True)
    piv_train = train.shape[0]
    df_all = pd.concat((train, test), axis=0, ignore_index=True)

    #################
    # Custom Fillna
    #################
    df_all['cited_n'].fillna(0., inplace=True)
    df_all['cited_nmiss'].fillna(0., inplace=True)
    df_all['cited_age_std'].fillna(0., inplace=True)
    # Goal : Replace missing values in COUNTRY with the most recurrent value associated in FISRT_APP_COUNTRY
    # made with Jeremie Guez
    df_country_replacement = df_all.groupby('FISRT_APP_COUNTRY')['COUNTRY'].value_counts().reset_index()
    idx_max = df_country_replacement.groupby(['FISRT_APP_COUNTRY'])[0].transform(max) == df_country_replacement[0]
    df_country_replacement = df_country_replacement[idx_max].drop_duplicates('FISRT_APP_COUNTRY')
    dict_to_replace = df_country_replacement.set_index('FISRT_APP_COUNTRY')['COUNTRY'].to_dict()
    df_all['COUNTRY'][df_all['COUNTRY'].isnull()] = df_all['FISRT_APP_COUNTRY'][df_all['COUNTRY'].isnull()].replace(dict_to_replace)

    ##################
    # Working on date
    ##################
    date_columns = ['PRIORITY_MONTH', 'FILING_MONTH', 'PUBLICATION_MONTH', 'BEGIN_MONTH']

    def to_timestamp(date_string):
        date_format = "%m/%Y"
        if date_string is np.nan:
            return None
        return calendar.timegm(datetime.datetime.strptime(str(date_string), date_format).utctimetuple())

    df_all['diff_publication_begin'] = df_all['PUBLICATION_MONTH'].map(to_timestamp) - df_all['BEGIN_MONTH'].map(to_timestamp)
    df_all['diff_filling_priority'] = df_all['FILING_MONTH'].map(to_timestamp) - df_all['PRIORITY_MONTH'].map(to_timestamp)

    regex = r'([\d]{2})/([\d]{4})'
    for col in date_columns:
        df_all[[str(col), str(col) + '_YEAR']] = df_all[col].str.extract(regex)

    #######################
    # Creating some columns
    #######################
    df_all['BEGIN_IS_FILING'] = 0
    df_all['BEGIN_IS_FILING'][df_all['BEGIN_MONTH'] == df_all['FILING_MONTH']] = 1

    col_classes = 'FIRST_CLASSE'
    df_all['class_lvl1'] = df_all[col_classes].str.extract(r'^([A-Z]{1})')
    df_all['class_lvl2'] = df_all[col_classes].str.extract(r'^([A-Z]{1}[0-9]{2})')
    df_all['class_lvl3'] = df_all[col_classes].str.extract(r'^([A-Z]{1}[0-9]{2}[A-Z]{1})')
    df_all['class_lvl4'] = df_all[col_classes].str.extract(r'^(.*?)/')
    df_all.drop('MAIN_IPC', axis=1, inplace=True)

    ########################
    # Take care of last NaN
    ########################
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    non_numeric_columns = df_all.select_dtypes(exclude=numerics).columns.difference(date_columns)

    df_all[non_numeric_columns] = df_all[non_numeric_columns].apply(LabelEncoder().fit_transform)

    all_data = Imputer().fit_transform(df_all)

    # Recreate train / test
    train, test = all_data[:piv_train], all_data[piv_train:]

    # Format target
    target = target.replace(to_replace=['GRANTED', 'NOT GRANTED'], value=[1, 0])

    # Save
    np.savetxt('train.txt', train, fmt='%s')
    np.savetxt('test.txt', test, fmt='%s')
    np.savetxt('target.txt', target, fmt='%s')


class MyXGB(BaseEstimator, ClassifierMixin):
    """Scikit-Learn wrapper for XGBoost."""

    def __init__(self, param, num_rounds):
        self.param = param
        self.num_rounds = num_rounds

    def fit(self, X, y):
        dx = xgb.DMatrix(X, label=y)
        estimator_ = xgb.train(self.param, dx, self.num_rounds)
        self.estimator_ = estimator_
        self.feature_importances_ = np.array(estimator_.get_fscore().values())
        return self

    def predict_proba(self, X):
        dx = xgb.DMatrix(X)
        return self.estimator_.predict(dx)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)


def _create_selector():
    train, target, _ = get_preproc()

    param = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'objective': 'multi:softprob',
             'lambda': 1, 'subsample': 0.9, 'eval_metric': 'auc', 'num_class': 2}
    num_rounds = 150
    rfecv = RFECV(estimator=MyXGB(param, num_rounds), cv=StratifiedKFold(target, 2), scoring='roc_auc')
    rfecv.fit(train, target)

    print("Optimal number of features : %d" % rfecv.n_features_)

    np.savetxt('feature_selector_support.txt', rfecv.support_, fmt='%s')

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
