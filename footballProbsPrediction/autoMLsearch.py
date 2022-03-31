from flaml import AutoML
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from xgboost import XGBClassifier
from sklearn.metrics import log_loss,accuracy_score


import numpy as np
import pandas as pd
import datetime as dt
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
import os, pickle, time

path = 'C:\\Users\\gcram\\Documents\\Datasets\\football-match-probability-prediction\\'
outfile = os.path.join(path, f'footbal_train_processed.pkl')
X = pd.read_pickle(outfile)
y = X['target'].replace({'home': 1, 'draw': 2,'away':3})
X.drop('target',axis = 1,inplace = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
automl = AutoML()
settings = {
    "time_budget": 600,  # total running time in seconds
    "metric": 'log_loss',  # can be: 'r2', 'rmse', 'mae', 'mse', 'accuracy', 'roc_auc', 'roc_auc_ovr',
                           # 'roc_auc_ovo', 'log_loss', 'mape', 'f1', 'ap', 'ndcg', 'micro_f1', 'macro_f1'
    "task": 'classification',  # task type
    "log_file_name": 'airlines_experiment.log',  # flaml log file
    "seed": 7654321,    # random seed
}

automl.fit(X_train=X_train, y_train=y_train, **settings)

'''retrieve best config and best learner'''
print('Best ML leaner:', automl.best_estimator)
print('Best hyperparmeter config:', automl.best_config)
print('Best accuracy on validation data: {0:.4g}'.format(1-automl.best_loss))
print('Training duration of best run: {0:.4g} s'.format(automl.best_config_train_time))

'''pickle and save the automl object'''
import pickle
with open('automl.pkl', 'wb') as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
    
    
    
    
# path = 'C:\\Users\\gcram\\Documents\\Datasets\\football-match-probability-prediction\\'
# outfile = os.path.join(path, f'footbal_train_processed.npz')
# # df = simplePreProcess(path,stage = 'train')
# # df.to_pickle(outfile)
#
# X,y = process_data(path,stage = 'train')
# np.savez(outfile, X=X,y=y)