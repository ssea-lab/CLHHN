import torch
from torch import nn
import numpy as np
from src.layers.readout import *
from src.layers.hgnn import *
import torch.nn.functional as F


class LightCLHHN(nn.Module):
	def __init__(self, config):
		super(LightCLHHN, self).__init__()
		self.item_num = config['item_num']
		self.hidden_size = config['hidden_size']
		self.embedding = nn.Embedding(config['item_num'] + config['category_num'], self.hidden_size, padding_idx=0)
		self.item_cates = torch.tensor(config['item_cates'], dtype=torch.long, device=config['device'])

		self.hgnn = WLHGAConvWeight(self.hidden_size, config['step'], config['hg_dropout'])

		self.session_readout = LastAttention(self.hidden_size * 2)

		self.item_dropout = nn.Dropout(config['dropout'])
		self.loss = nn.CrossEntropyLoss()
		self._reset_parameters()

	def _reset_parameters(self):
		stdv = 1.0 / np.sqrt(self.hidden_size)
		for weight in self.parameters():
			weight.data.uniform_(-stdv, stdv)

	def nodes2items(self, nodes_hidden, alias):
		get = lambda i: nodes_hidden[i][alias[i]]
		seq_hidden = torch.stack([get(i) for i in torch.arange(len(alias)).long()])
		return seq_hidden

	def get_scores(self, inputs):
		item_seq, label = inputs[0:2]                                           # (b, l)    (b)
		mask = item_seq.gt(0)
		nodes, hn_adj, alias_item, alias_cate = inputs[2:6]     # (b, node_num) (b, edge_num, node_num) (b, l) (b, l)

		nodes_hidden_in = self.embedding(nodes)                                 # (b, node_num, h)
		nodes_hidden_in = F.normalize(nodes_hidden_in, p=2, dim=-1)
		nodes_hidden_in = self.item_dropout(nodes_hidden_in)

		# Weighted Lossless HyperGraph Attention Convolutional Neural Network
		nodes_hidden, edges_hidden = self.hgnn(nodes_hidden_in, hn_adj)

		item_seq_hidden = self.nodes2items(nodes_hidden, alias_item)
		cate_seq_hidden = self.nodes2items(nodes_hidden, alias_cate)

		seq_hidden = torch.cat([item_seq_hidden, cate_seq_hidden], dim=-1)
		seq_hidden = F.normalize(seq_hidden, p=2, dim=-1)
		seq_output = self.session_readout(seq_hidden, mask)

		item_emb = self.embedding.weight[:self.item_num, :]
		cates_emb = self.embedding(self.item_cates)
		item_emb = torch.cat([item_emb, cates_emb], dim=-1)

		seq_output = F.normalize(seq_output, p=2, dim=-1)
		item_emb = F.normalize(item_emb, p=2, dim=-1)
		# item_emb = self.item_dropout(item_emb)
		scores = torch.matmul(seq_output, item_emb.transpose(0, 1))
		scores = scores * 16
		return scores

	def cal_loss(self, inputs):
		labels = inputs[1]
		scores = self.get_scores(inputs)
		scores = scores[:, 1:]
		loss = self.loss(scores, labels - 1)
		return loss

	def predict(self, inputs, phase):
		scores = self.get_scores(inputs)
		scores = scores[:, 1:]

		return scores

