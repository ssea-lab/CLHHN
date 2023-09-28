import pickle
import random
from src.data.data import *
from torch.utils.data import DataLoader


def init_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True


def get_item_num(dataset_name):
	dataset2item_num = {'diginetica': 43097 + 1,
	                    'tmall': 40727 + 1,
	                    '2019-oct': 27480 + 1
	                    }
	return dataset2item_num[dataset_name]


def get_cate_num(dataset_name):
	dataset2cate_num = {'diginetica': 995 + 1,
	                    'tmall': 711 + 1,
	                    '2019-oct': 448 + 1
	                    }
	return dataset2cate_num[dataset_name]


def get_dataset(config):
	config['logger'].info("get_dataset...")
	data_path = '../../datasets/' + config['dataset'] + '/idata.txt'
	with open(data_path, 'rb') as f:
		data_dicts = pickle.load(f)
	train_dict, test_dict, maps = data_dicts
	train_ds = CLHHNDataset(config, train_dict)
	test_ds = CLHHNDataset(config, test_dict)
	return train_ds, test_ds, maps


def get_dataloader(config, datasets):
	config['logger'].info("get_dataloader...")
	train_ds, test_ds = datasets
	drop_last = False
	if torch.cuda.is_available():
		train_dl = DataLoader(dataset=train_ds, batch_size=config['batch_size'],
		                      shuffle=True, num_workers=4, pin_memory=True, drop_last=drop_last)
		test_dl = DataLoader(dataset=test_ds, batch_size=config['batch_size'],
		                     shuffle=False, num_workers=4, pin_memory=True, drop_last=drop_last)
	else:
		train_dl = DataLoader(dataset=train_ds, batch_size=config['batch_size'],
		                      shuffle=False, num_workers=1, pin_memory=False, drop_last=drop_last)
		test_dl = DataLoader(dataset=test_ds, batch_size=config['batch_size'],
		                     shuffle=False, num_workers=1, pin_memory=False, drop_last=drop_last)
	return train_dl, test_dl


def early_stopping(curr_result, best_result, cur_step, patience):
	update_flag, stop_flag = False, False
	if curr_result > best_result:
		cur_step = 0
		best_result = curr_result
		update_flag = True
	else:
		cur_step += 1
		if cur_step >= patience:
			stop_flag = True
	return best_result, cur_step, update_flag, stop_flag


def print_result(config, metric_results, topks):
	hits, mrrs, ndcgs = metric_results
	width, style = '{:<15}', '%.3f'
	# print('-' * 110)
	# print('|', end='')
	for idx, topk in enumerate(topks):
		# print('|',
		#       width.format(style % hits[idx]),
		#       width.format(style % mrrs[idx]),
		#       width.format(style % ndcgs[idx]),
		#       '|', end=''
		#       )
		curr_result = '  |' + \
		              width.format(style % hits[idx]) + \
		              width.format(style % mrrs[idx]) + \
		              width.format(style % ndcgs[idx]) + \
		              '|'
		config['logger'].info(curr_result)
	# print('|')
	# print('||', width.format(style % his_dcg),
	#       width.format(style % his_mrr),
	#       )
	# print('-' * 110)

