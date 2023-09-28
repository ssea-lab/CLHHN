import numpy as np
import torch


def get_sample_data(data_dict, config):
	if config['sample'] != -1:
		for col in data_dict:
			data_dict[col] = data_dict[col][:config['sample']]
	return data_dict


def get_max_seq_len(data_dict, config):
	if config['max_seq_len'] == -1:
		iseq_lens = [len(iseq) for iseq in data_dict['item_seq']]
		max_seq_len = max(iseq_lens)
	else:
		max_seq_len = config['max_seq_len']
	return max_seq_len


# ============================ align sequence =================================
def align_seq_category(data_dict, max_seq_len, config):
	reverse = config['seq_reverse']
	item_num = config['item_num']

	datasize = len(data_dict['item_seq'])
	item_seqs, cate_seqs = data_dict['item_seq'], data_dict['category_seq']
	cate_seqs = [[cate_id + item_num for cate_id in cate_seq] for cate_seq in cate_seqs]

	labels = [item_seq[-1] for item_seq in item_seqs]
	item_seqs = [item_seq[:-1] for item_seq in item_seqs]
	categories = [cate_seq[-1] for cate_seq in cate_seqs]
	cate_seqs = [cate_seq[:-1] for cate_seq in cate_seqs]
	seq_lens = [max_seq_len if len(i_seq) > max_seq_len else len(i_seq) for i_seq in item_seqs]
	new_i_seqs, new_c_seqs = [], []
	for idx in range(datasize):
		i_seq, c_seq = item_seqs[idx], cate_seqs[idx]
		if reverse:
			i_seq, c_seq = list(reversed(i_seq)), list(reversed(c_seq))
		if len(i_seq) < max_seq_len:
			new_i_seq = i_seq + [0] * (max_seq_len - len(i_seq))
			new_c_seq = c_seq + [0] * (max_seq_len - len(c_seq))
		else:
			new_i_seq = i_seq[:max_seq_len] if reverse else i_seq[-max_seq_len:]
			new_c_seq = c_seq[:max_seq_len] if reverse else c_seq[-max_seq_len:]
		new_i_seqs += [new_i_seq]
		new_c_seqs += [new_c_seq]

	new_data_dict = {'item_seq': new_i_seqs, 'label': labels,
	                 'category_seq': new_c_seqs, 'category': categories,
	                 'seq_len': seq_lens}
	return new_data_dict


def get_item_cate_hypergraph(item_seq, cate_seq, seq_len, max_seq_len, config):
	window_size_config = config['window_size']
	window_size = min(window_size_config, seq_len)

	max_n_node = max_seq_len * 2
	# context; session; unit
	context_edge_num = np.sum(np.arange(max_seq_len + 1)) - np.sum(np.arange(max_seq_len - window_size_config + 1))
	session_edge_num = 1
	unit_edge_num = max_seq_len

	max_n_edge = 0
	max_n_edge += context_edge_num * 2
	max_n_edge += session_edge_num * 2
	max_n_edge += unit_edge_num

	hn_adj = np.zeros((max_n_edge, max_n_node))
	nodes = np.unique(item_seq + cate_seq)

	alias_item = [np.where(nodes == item_id)[0][0] for item_id in item_seq]
	alias_cate = [np.where(nodes == cate_id)[0][0] for cate_id in cate_seq]
	nodes = list(nodes) + [0] * (max_n_node - len(nodes))
	item_node_seq = alias_item[:seq_len]
	cate_node_seq = alias_cate[:seq_len]

	# context
	edge_nodes = []
	for idx in range(seq_len):
		for win_size in range(1, window_size + 1):
			end = idx + win_size
			if end > seq_len:
				break
			edge_nodes += [item_node_seq[idx: end]]
			edge_nodes += [cate_node_seq[idx: end]]
	# session
	edge_nodes += [item_node_seq]
	edge_nodes += [cate_node_seq]
	# unit
	for idx in range(seq_len):
		edge_nodes += [[item_node_seq[idx], cate_node_seq[idx]]]

	# weighted lossless hypergraph construction
	for edge_idx in range(len(edge_nodes)):
		for node_id in edge_nodes[edge_idx]:
			hn_adj[edge_idx, node_id] += 1

	output = nodes, hn_adj, alias_item, alias_cate
	return output


# ============================ other utils ========================
def to_tensor_long(element_tuple):
	temp_list = (torch.tensor(element).long() for element in element_tuple)
	return temp_list


def to_tensor_float(element_tuple):
	temp_list = (torch.tensor(element).float() for element in element_tuple)
	return temp_list
