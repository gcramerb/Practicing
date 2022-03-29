import numpy as np
import pandas as pd
import datetime as dt
import os, pickle, time
from tqdm import tqdm

"""
Possible handcrafted Features on 10 previous match:

Num victoris
num draws
num losses
num matchs the same coach
num matchs that the expected had occurred: ex. the highest rating team won.


"""


def process_data(path,stage = 'train'):
	"""
	train
	test
	sample_submission
	"""
	# loading the data
	df = pd.read_csv(os.path.join(path,f'{stage}.csv'))

	MASK = -1  # fill NA with -1
	T_HIST = 10  # time history, last 10 games
	# for cols "date", change to datatime
	for col in df.filter(regex='date', axis=1).columns:
		df[col] = pd.to_datetime(df[col])
	# Creating some feature engineering
	print('processing hitorical data...')
	for i in tqdm(range(1, 11)):  # range from 1 to 10
		# Feat. difference of days
		df[f'home_team_history_match_DIFF_day_{i}'] = (df['match_date'] - df[f'home_team_history_match_date_{i}']).dt.days
		df[f'away_team_history_match_DIFF_days_{i}'] = (df['match_date'] - df[f'away_team_history_match_date_{i}']).dt.days
		# Feat. difference of scored goals
		df[f'home_team_history_DIFF_goal_{i}'] = df[f'home_team_history_goal_{i}'] - df[f'home_team_history_opponent_goal_{i}']
		df[f'away_team_history_DIFF_goal_{i}'] = df[f'away_team_history_goal_{i}'] - df[f'away_team_history_opponent_goal_{i}']
		# Results: multiple nested where
		df[f'home_team_result_{i}'] = np.where(df[f'home_team_history_DIFF_goal_{i}'] > 0, 1,
		                                       (np.where(df[f'home_team_history_DIFF_goal_{i}'] == 0, 0,
		                                        np.where(df[f'home_team_history_DIFF_goal_{i}'].isna(),np.nan, -1))))
		
		df[f'away_team_result_{i}'] = np.where(df[f'away_team_history_DIFF_goal_{i}'] > 0, 1,
		                                       (np.where(df[f'away_team_history_DIFF_goal_{i}'] == 0, 0,
		                                        np.where(df[f'away_team_history_DIFF_goal_{i}'].isna(), np.nan, -1))))
		
		# Feat. difference of rating ("modified" ELO RATING)
		df[f'home_team_history_ELO_rating_{i}'] = 1 / (1 + 10 ** ((df[f'home_team_history_opponent_rating_{i}'] - df[f'home_team_history_rating_{i}']) / 10))
		df[f'away_team_history_ELO_rating_{i}'] = 1 / (1 + 10 ** ((df[f'away_team_history_opponent_rating_{i}'] - df[f'away_team_history_rating_{i}']) / 10))
		# df[f'away_team_history_DIFF_rating_{i}'] =  - df[f'away_team_history_opponent_rating_{i}']
		# Feat. same coach id
		df[f'home_team_history_SAME_coaX_{i}'] = np.where(df['home_team_coach_id'] == df[f'home_team_history_coach_{i}'], 1, 0)
		df[f'away_team_history_SAME_coaX_{i}'] = np.where(df['away_team_coach_id'] == df[f'away_team_history_coach_{i}'], 1, 0)
		# Feat. same league id
		#df[f'home_team_history_SAME_leaG_{i}'] = np.where(df['league_id'] == df[f'home_team_history_league_id_{i}'],1, 0)
		#df[f'away_team_history_SAME_leaG_{i}'] = np.where(df['league_id'] == df[f'away_team_history_league_id_{i}'],1, 0)
	# Fill NA with -1
	print('done')
	df.fillna(MASK, inplace=True)
	
	# save targets
	# y_train = train[['target_int']].to_numpy().reshape(-1, 1)
	id = df['id'].copy()
	y = df['target'].copy()
	# keep only some features
	df.drop(['id', 'target', 'home_team_name', 'away_team_name'], axis=1, inplace=True)
	df['is_cup'] = df['is_cup'].replace({True: 1, False: 0})
	# Exclude all date, league, coach columns
	df.drop(df.filter(regex='date').columns, axis=1, inplace=True)
	df.drop(df.filter(regex='league').columns, axis=1, inplace=True)
	df.drop(df.filter(regex='coach').columns, axis=1, inplace=True)
	
	# from sklearn.preprocessing import LabelEncoder
	# le = LabelEncoder()
	# df_train['home_team_name'] = le.fit_transform(df_train['home_team_name'])
	# df_train['away_team_name'] = le.fit_transform(df_train['away_team_name'])
	# df_train['league_name'] = le.fit_transform(df_train['league_name'])
	
	# Store feature names
	feature_names = list(df.columns)
	# Scale features using statistics that are robust to outliers
	# RS = RobustScaler()
	# train = RS.fit_transform(train)

	# Back to pandas.dataframe
	df = pd.DataFrame(df, columns=feature_names)
	df = pd.concat([id, df], axis=1)

	# Pivot dataframe to create an input array for the LSTM network
	feature_groups = ["home_team_history_is_play_home", "home_team_history_is_cup",
	                  "home_team_history_goal", "home_team_history_opponent_goal",
	                  "home_team_history_rating", "home_team_history_opponent_rating",
	                  "away_team_history_is_play_home", "away_team_history_is_cup",
	                  "away_team_history_goal", "away_team_history_opponent_goal",
	                  "away_team_history_rating", "away_team_history_opponent_rating",
	                  "home_team_history_match_DIFF_day", "away_team_history_match_DIFF_days",
	                  "home_team_history_DIFF_goal", "away_team_history_DIFF_goal",
	                  "home_team_history_ELO_rating", "away_team_history_ELO_rating",
	                  "home_team_history_SAME_coaX", "away_team_history_SAME_coaX",
	                  "home_team_history_SAME_leaG", "away_team_history_SAME_leaG",
	                  "home_team_result", "away_team_result"]
	# Pivot dimension (id*features) x time_history
	x_pivot = pd.wide_to_long(df, stubnames=feature_groups,i='id', j='time', sep='_', suffix='\d+')
	
	# Trying to keep the same id order
	x = pd.merge(id, x_pivot, on="id")
	x = x.drop(['id'], axis=1).to_numpy().reshape(-1, T_HIST, x_pivot.shape[-1])

	return x,y

def simplePreProcess(path,stage = 'train'):
	df = pd.read_csv(os.path.join(path, f'{stage}.csv'))

	MASK = -1  # fill NA with -1
	T_HIST = 10  # time history, last 10 games
	# for cols "date", change to datatime
	for col in df.filter(regex='date', axis=1).columns:
		df[col] = pd.to_datetime(df[col])
	# Creating some feature engineering
	print('processing hitorical data...')
	
	hist_sum_columns = ['home_team_history_goals_scored','home_team_history_goals_against','home_team_history_goals_balance',
	                    'away_team_history_goals_scored','away_team_history_goals_against','away_team_history_goals_balance',
	                    'home_team_hist_win','home_team_hist_loss','home_team_hist_draw',
	                    'away_team_hist_win','away_team_hist_loss','away_team_hist_draw',
	                    'home_team_num_expected_result','away_team_num_expected_result']
	df[hist_sum_columns] = 0
	for i in tqdm(range(1, 11)):  # range from 1 to 10
		# Feat. difference of scored goals
		for team in ['home','away']:
			df[f'{team}_team_history_goals_scored'] += df[f'{team}_team_history_goal_{i}']
			df[f'{team}_team_history_goals_against'] += df[f'{team}_team_history_opponent_goal_{i}']
	
			# Results: multiple nested where
			df[f'{team}_team_hist_win'] += np.where(df[f'{team}_team_history_goal_{i}'] > df[f'{team}_team_history_opponent_goal_{i}'], 1, 0)
			df[f'{team}_team_hist_loss'] += np.where(df[f'{team}_team_history_goal_{i}'] < df[f'{team}_team_history_opponent_goal_{i}'], 1, 0)
			df[f'{team}_team_hist_draw'] += np.where(df[f'{team}_team_history_goal_{i}'] == df[f'{team}_team_history_opponent_goal_{i}'], 1, 0)
	
			# Feat. difference of rating ("modified" ELO RATING)
			a = np.where(df[f'{team}_team_history_opponent_rating_{i}'] > df[f'{team}_team_history_rating_{i}'],0.5,0)
			b = np.where(df[f'{team}_team_history_goal_{i}'] > df[f'{team}_team_history_opponent_goal_{i}'],0.5,0)
			df[f'{team}_team_num_expected_result'] += np.add(a,b).astype('int')

	
	df[f'home_team_history_goals_balance'] += df[f'home_team_history_goals_scored'] - df[
		f'home_team_history_goals_against']
	df[f'away_team_history_goals_balance'] += df[f'away_team_history_goals_scored'] - df[
		f'away_team_history_goals_against']
	print('done')
	df.fillna(MASK, inplace=True)
	
	# save targets

	id = df['id'].copy()

	# keep only some features
	df.drop(['id', 'target', 'home_team_name', 'away_team_name'], axis=1, inplace=True)
	df['is_cup'] = df['is_cup'].replace({True: 1, False: 0})
	# Exclude all date, league, coach columns
	df.drop(df.filter(regex='date').columns, axis=1, inplace=True)
	df.drop(df.filter(regex='league').columns, axis=1, inplace=True)
	df.drop(df.filter(regex='coach').columns, axis=1, inplace=True)
	for i in range(1,10):
		df.drop(df.filter(regex=f'_{i}').columns, axis=1, inplace=True)

	# Store feature names
	feature_names = list(df.columns)
	# Scale features using statistics that are robust to outliers
	RS = RobustScaler()
	df = RS.fit_transform(df)
	
	# Back to pandas.dataframe
	df = pd.DataFrame(df, columns=feature_names)
	df = pd.concat([id, df], axis=1)
	return df
	


path = 'C:\\Users\\gcram\\Documents\\Datasets\\football-match-probability-prediction\\'
df = simplePreProcess(path,stage = 'train')
#X,y = process_data(path,stage = 'train')

outfile = os.path.join(path, f'footbal_{stage}_processed')
np.savez(outfile, X=X,y=y)


from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import Dataset,DataLoader


class myDataset(Dataset):
	"""
	Class that recives the already processed data.
	"""
	def __init__(self,x,y):
		self.x, self.y = x,y
	def __len__(self):
		return len(self.y)
	def __getitem__(self, index):
		return self.x[index],self.y[index]

class MyDataModule(LightningDataModule):
	def __init__(self ,batch_size, path):
		super().__init__()
		self.batch_size = batch_size
		self.path_file = path
		self.dataset = {}

	def _setup(self):
		for stage in ['train']:
			outfile = os.path.join(self.path_file,f'footbal_{stage}_processed.npz')
			
			with np.load(outfile,allow_pickle=True) as tmp:
				X = tmp['X']
				y = tmp['y']
			# split Train test
			raise ValueError('Split train test')
			self.dataset[stage] = myDataset(X, y)

	def train_dataloader(self):
		return DataLoader(self.dataset['train'],
		                  drop_last=True,
		                  batch_size=self.batch_size)
	def val_dataloader(self):
		return DataLoader(self.dataset['test'],
		                  drop_last= True,
		                  batch_size=self.batch_size)
	def test_dataloader(self):
		return DataLoader(self.dataset['test'],
		                  drop_last= True,
		                  batch_size=self.batch_size)
	