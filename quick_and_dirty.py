"""
Quick and Dirty model. Give 0.7147 on public leaderboard.

@author Yann Carbonne

Basic preprocessing with LabelEncoder and Imputer.
Simple XGBoost as training and predict model.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, Imputer

# Import
test = pd.read_csv('test.csv', sep=';', na_values='(MISSING)')
train = pd.read_csv('train.csv', sep=';', na_values='(MISSING)')
target = train['VARIABLE_CIBLE']
train.drop('VARIABLE_CIBLE', axis=1, inplace=True)
piv_train = train.shape[0]

# Creating a DataFrame with train+test data
df_all = pd.concat((train, test), axis=0, ignore_index=True)


#################
# Preprocessing
#################
numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
date_columns = np.array(['PRIORITY_MONTH', 'FILING_MONTH', 'PUBLICATION_MONTH', 'BEGIN_MONTH'])
non_numeric_columns = train.select_dtypes(exclude=numerics).columns.difference(date_columns)

# Label Encoding
df_all[non_numeric_columns] = df_all[non_numeric_columns].apply(LabelEncoder().fit_transform)

# Just get the year
regex = r'/([\d]{4})'
for col in date_columns:
    df_all[col] = df_all[col].str.extract(regex)

# Replacing NaN values with mean
all_data = Imputer().fit_transform(df_all)

# Recreate train / test
train, test = all_data[:piv_train], all_data[piv_train:]

# Format target
target = target.replace(to_replace=['GRANTED', 'NOT GRANTED'], value=[1, 0])


##################
# Train & Predict
##################
dtrain = xgb.DMatrix(train, label=target)
dtest = xgb.DMatrix(test)
param = {'max_depth': 10, 'eta': 0.1, 'silent': 1, 'objective': 'binary:logistic',
         'lambda': 1, 'subsample': 0.9, 'eval_metric': 'auc'}
num_rounds = 150
bst = xgb.train(param, dtrain, num_rounds)
y_predict = bst.predict(dtest)
np.savetxt('test_preproc.txt', y_predict, fmt='%s')
