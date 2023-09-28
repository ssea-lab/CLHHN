from torch.utils.data import Dataset
from src.utils.data_utils import *


class CLHHNDataset(Dataset):
	def __init__(self, config, data_dict):
		self.config = config
		self.data_dict = data_dict
		self.sample_data()
		self.max_seq_len = get_max_seq_len(data_dict, config)
		self.align_seq()
		self.window_size = self.config['window_size']

	def sample_data(self):
		self.data_dict = get_sample_data(data_dict=self.data_dict, config=self.config)

	def align_seq(self):
		self.data_dict = align_seq_category(data_dict=self.data_dict,
		                                    max_seq_len=self.max_seq_len,
		                                    config=self.config)

	def __getitem__(self, idx):
		item_seq, label = self.data_dict['item_seq'][idx], self.data_dict['label'][idx]
		category_seq = self.data_dict['category_seq'][idx]
		seq_len = self.data_dict['seq_len'][idx]

		nodes, hn_adj, alias_item, alias_cate = get_item_cate_hypergraph(item_seq, category_seq, seq_len,
		                                                                 self.max_seq_len, self.config)

		long_tuple = item_seq, label, nodes, alias_item, alias_cate, seq_len
		float_tuple = hn_adj,
		item_seq, label, nodes, alias_item, alias_cate, seq_len = to_tensor_long(long_tuple)
		hn_adj, = to_tensor_float(float_tuple)

		sample = item_seq, label, nodes, hn_adj, alias_item, alias_cate, seq_len
		return sample

	def __len__(self):
		return len(self.data_dict['item_seq'])
