import numpy as np


class Evaluator:
	def __init__(self, config):
		self.config = config
		self.topks = config['topk_list']
		self.item_seqs, self.labels = None, None
		self.ranks, self.label_ranks = None, None
	
	def eval(self, item_seqs, labels, ranks):
		self.item_seqs, self.labels = item_seqs.cpu().detach().numpy(), labels.cpu().detach().numpy()
		self.ranks = ranks.cpu().detach().numpy()
		self.label_ranks = self.ranks[np.arange(len(self.ranks)), self.labels - 1]
		hits, mrrs, ndcgs = self.get_hit(), self.get_mrr(), self.get_ndcg()
		return hits, mrrs, ndcgs
		
	def get_hit(self):
		hits = [[] for _ in range(len(self.topks))]
		for rank in self.label_ranks:
			for idx, topk in enumerate(self.topks):
				if rank < topk:
					hits[idx] += [1]
				else:
					hits[idx] += [0]
		return hits
	
	def get_mrr(self):
		mrrs = [[] for _ in range(len(self.topks))]
		for rank in self.label_ranks:
			for idx, topk in enumerate(self.topks):
				if rank < topk:
					mrrs[idx] += [1 / (rank + 1)]
				else:
					mrrs[idx] += [0]
		return mrrs
	
	def get_ndcg(self):
		ndcgs = [[] for _ in range(len(self.topks))]
		for rank in self.label_ranks:
			for idx, topk in enumerate(self.topks):
				if rank < topk:
					ndcgs[idx] += [1 / np.log2(rank + 1 + 1)]
				else:
					ndcgs[idx] += [0]
		return ndcgs
		
		
		
