from torch import nn
import torch


class lstmModel(nn.Module):
	def __init__(self,
	             n_classes=3,
	             stacked_layers=2,
	             hidden_size=64,
	             embedding_dim=300,
	             batch_size=512,
	             seq_len1=11,
	             seq_len2=6):
		super(lstmModel, self).__init__()
		
		self.hidden_size = hidden_size
		self.bidirectional = True
		self.n_classes = n_classes
		self.stacked_layers = stacked_layers
		self.embedding_dim = embedding_dim
		self.seq_len1 = seq_len1
		self.seq_len2 = seq_len2
		self.batch_size = batch_size
	
	def build(self, emb_matrix):
		num_embeddings, embedding_dim = len(emb_matrix), len(emb_matrix[0])
		weight = torch.FloatTensor(emb_matrix)
		self.embedding = nn.Embedding.from_pretrained(weight, freeze=True, padding_idx=0)
		
		self.lstm_hypotheses = nn.LSTM(input_size=self.embedding_dim,
		                               hidden_size=self.hidden_size,
		                               num_layers=self.stacked_layers,
		                               batch_first=True,
		                               dropout=0.2,
		                               bidirectional=self.bidirectional)
		self.lstm_evidences = nn.LSTM(input_size=self.embedding_dim,
		                              hidden_size=self.hidden_size,
		                              num_layers=self.stacked_layers,
		                              batch_first=True,
		                              dropout=0.2,
		                              bidirectional=self.bidirectional)
		
		self.lstm_agg = nn.LSTM(input_size=2 * self.hidden_size,
		                        hidden_size=self.hidden_size,
		                        num_layers=self.stacked_layers,
		                        batch_first=True,
		                        dropout=0.2,
		                        bidirectional=self.bidirectional)
		
		self.FC = nn.Sequential(
			nn.Flatten(),  # 4352 ?
			nn.Linear(2 * (self.seq_len1 + self.seq_len2) * self.hidden_size, 256),
			nn.ReLU(),
			nn.Linear(256, 128),
			nn.ReLU(),
			nn.Linear(128, 32),
			nn.ReLU(),
			nn.Linear(32, self.n_classes),
			nn.Sigmoid()
		)
	
	def forward(self, input):
		hypotheses, evidences = input[0], input[1]
		hypotheses, evidences = self.embedding(hypotheses), self.embedding(evidences)
		hypotheses, (hH, ch) = self.lstm_hypotheses(hypotheses)
		evidences, (hE, cE) = self.lstm_evidences(evidences, (hH, ch))
		comb_outputs = torch.cat((hypotheses, evidences), 1)
		out, (hidden, _) = self.lstm_agg(comb_outputs, (hE, cE))
		probs = self.FC(out)
		return probs


#
# EPOCHS = 25
# BATCH_SIZE = 32
# EMBEDDING_SIZE = 300
# VOCAB_SIZE = len(vocab.word2index)
# TARGET_SIZE = len(tag2idx)
# LEARNING_RATE = 0.005

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch import optim

import sys, os, argparse, pickle
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score

sys.path.insert(0, '../')

from models.model import lstmModel
# import geomloss

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.trainer.supporters import CombinedLoader
from collections import OrderedDict

"""
There is tw encoders that train basecally the same thing,
in the future I can use only one Decoder (more dificult to converge)
"""


class NLItrainer(LightningModule):
	
	def __init__(
			self,
			lr: float = 0.002,
			seq_len1: int = 11,
			seq_len2: int = 6,
			path_emb: str = None,
			**kwargs
	):
		super().__init__()
		self.save_hyperparameters()
		if path_emb:
			emb_matrix = []
			file = os.path.join(path_emb, f'Vocab.npz')
			with np.load(file, allow_pickle=True) as tmp:
				emb_matrix = tmp['Vocab']
		
		# networks
		self.model = lstmModel(seq_len1=seq_len1, seq_len2=seq_len2)
		self.model.build(emb_matrix)
		self.loss = torch.nn.CrossEntropyLoss()
	
	def forward(self, X):
		return self.model(X)
	
	def _shared_eval_step(self, batch, stage='val'):
		sent1, sent2, label = batch[0], batch[1], batch[2].long()
		pred = self.model((sent1, sent2))
		if stage == 'val':
			loss = self.loss(pred, label)
			acc = accuracy_score(label.cpu().numpy(), np.argmax(pred.cpu().numpy(), axis=1))
			metrics = {'val_acc': acc,
			           'val_loss': loss.detach()}
		
		elif stage == 'test':
			acc = accuracy_score(label.cpu().numpy(), np.argmax(pred.cpu().numpy(), axis=1))
			metrics = {'test_acc': acc}
		return metrics
	
	def training_step(self, batch, batch_idx):
		
		sent1, sent2, label = batch[0], batch[1], batch[2].long()
		pred = self.model((sent1, sent2))
		loss = self.loss(pred, label)
		tqdm_dict = {f"train_loss": loss}
		output = OrderedDict({"loss": loss, "progress_bar": tqdm_dict})
		
		return output
	
	def predict(self, dl, stage='test'):
		outcomes = {}
		with torch.no_grad():
			true_list = []
			pred_list = []
			
			for batch in dl:
				sent1, sent2, label = batch[0], batch[1], batch[2].long()
				pred = self.model((sent1, sent2))
				true_ = label.cpu().numpy()
				pred_ = np.argmax(pred.cpu().numpy(), axis=1)
				true_list.append(true_)
				pred_list.append(pred_)
			
			outcomes[f'true_{stage}'] = np.concatenate(true_list, axis=0)
			outcomes[f'pred_{stage}'] = np.concatenate(pred_list, axis=0)
			return outcomes
	
	def validation_step(self, batch, batch_idx):
		# with torch.no_grad():
		metrics = self._shared_eval_step(batch, stage='val')
		return metrics
	
	def validation_epoch_end(self, out):
		keys_ = out[0].keys()
		metrics = {}
		for k in keys_:
			val = [i[k] for i in out]
			if 'loss' in k:
				metrics[k] = torch.mean(torch.stack(val))
			else:
				metrics[k] = np.mean(val)
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
	
	def test_step(self, batch, batch_idx):
		metrics = self._shared_eval_step(batch, stage='test')
		for k, v in metrics.items():
			self.log(k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True)
		return metrics
	
	def configure_optimizers(self):
		opt = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr)
		return [opt]
