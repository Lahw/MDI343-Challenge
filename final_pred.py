"""
Final prediction.

@author Yann Carbonne.

Give 0.7212 on the public leaderboard.
"""
from preprocessing import get_preproc
import pandas as pd
import xgboost as xgb
import numpy as np


def _get_indices(array, value, index_column=0):
    return np.where(array[:, index_column] == value)[0]

# get data
train, target, test = get_preproc(with_selector=True)

# get best param
best_param_df = pd.read_csv('hyperopt_results.csv')
best_param = best_param_df.sort_values('score', ascending=False).head(1).to_dict(orient='records')[0]
del best_param['Unnamed: 0']
del best_param['score']

# train & predict
final_pred = np.zeros(test.shape[0])
for value in np.unique(train[:, 0]):
    train_split = train[_get_indices(train, value)]
    target_split = target[_get_indices(train, value)]
    test_split = test[_get_indices(test, value)]
    dtrain = xgb.DMatrix(train_split, label=target_split)
    dtest = xgb.DMatrix(test_split)
    num_rounds = 5000
    bst = xgb.train(best_param, dtrain, num_rounds)
    final_pred[_get_indices(test, value)] = bst.predict(dtest)
np.savetxt('final_pred.txt', final_pred, fmt='%s')
