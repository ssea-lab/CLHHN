import a
import time
import torch
import argparse
from src.utils.utils import init_seed, get_item_num, get_dataset, get_dataloader, get_cate_num
from src.utils.log_utils import get_logger
from src.models.light_clhhn import *
from src.trainer.trainer import Trainer


def run(opt):
	config = {'seed': 2022,
	          'topk_list': [10, 20],
	          'model': 'CLHHN',

	          # data
	          'dataset': opt.dataset,   # tmall | diginetica | 2019-oct
	          'sample': opt.sample,     # -1 for all; n for n;
	          'seq_reverse': True,
	          'item_num': get_item_num(opt.dataset),
	          'category_num': get_cate_num(opt.dataset),
	          'max_seq_len': opt.msl,

	          # hypergraph
	          'window_size': 0,
	          'step': 0,
	          'hg_dropout': 0.1,

	          # train
	          'batch_size': 100,
	          'hidden_size': 100,
	          'dropout': 0.1,
	          'lr': 0.001,
	          'l2': 1e-5,
	          'lr_dc': 0.1,
	          'lr_dc_step': 3,

	          'epoch_num': 100,
	          'patience': 3,
	          'device': torch.device("cuda:{}".format(opt.gpu) if torch.cuda.is_available() else "cpu"),
	          'save': False if opt.save == 0 else True,
	          }
	log_file = "../logs/log_{}_{}.txt".format(config['dataset'], time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
	log_file = log_file.replace(":", "-").replace(" ", "_")
	config['logger'] = get_logger(log_file)

	if config['dataset'] == 'tmall':
		config['window_size'] = 5
		config['step'] = 1
	elif config['dataset'] == 'diginetica':
		config['window_size'] = 4
		config['step'] = 2
	elif config['dataset'] == '2019-oct':
		config['window_size'] = 6
		config['step'] = 1
	else:
		config['window_size'] = 5
		config['step'] = 1

	logger = config['logger']
	logger.info(config)
	init_seed(config['seed'])

	train_ds, test_ds, maps = get_dataset(config)
	dataloaders = get_dataloader(config, (train_ds, test_ds))

	item_cate_map = maps[0]
	item_cates = [item_cate_map[item_id] + config['item_num'] for item_id in range(config['item_num'])]
	config['item_cates'] = item_cates

	model = LightCLHHN(config).to(config['device'])
	config['logger'].info(model)

	trainer = Trainer(config, model)
	trainer.fit(dataloaders)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, default='tmall')
	parser.add_argument('--msl', type=int, default=10)
	parser.add_argument('--gpu', type=int, default=0)
	parser.add_argument('--sample', type=int, default=1000)
	parser.add_argument('--save', type=int, default=0)
	opt = parser.parse_args()
	run(opt)
