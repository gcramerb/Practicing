from sklearn.ensemble import RandomForestClassifier
from dataProcessing import simplePreProcess
import pandas as pd
import os

path = 'C:\\Users\\gcram\\Documents\\Datasets\\football-match-probability-prediction\\'
df = simplePreProcess(path,stage = 'test')
outfile = os.path.join(path, f'footbal_train_processed.pkl')
X = pd.read_pickle(outfile)
y = X['target'].replace({'home': 1, 'draw': 2,'away':3})
X.drop('target',axis = 1,inplace = True)

clf = RandomForestClassifier(n_estimators =  237,
                             max_features = 0.3353167384533355,
                             max_leaf_nodes = 338, criterion= 'entropy')
clf.fit(X,y)


probs = clf.predict_proba(df)
d
df_sub = pd.read_csv(os.path.join(path, f'sample_submission.csv'))
df_sub['home'] = probs[:,0]
df_sub['draw']= probs[:,1]
df_sub['away'] = probs[:,2]
df_sub.to_csv(os.path.join(path,'mySubmission.csv'))

