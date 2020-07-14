import torch
import random
import time
import os
import argparse
import sys
import numpy as np

from utils.misc import set_global_seeds, save_config, validate_config, check_device
from utils.dataset import Dataset
from models.Las import LAS
from trainer.trainer import Trainer


def load_arguments(parser):

	""" LAS """

	# paths
	parser.add_argument('--train_acous_path', type=str, default=None, help='train set acoustics')
	parser.add_argument('--dev_acous_path', type=str, default=None, help='dev set acoustics')
	parser.add_argument('--acous_norm_path', type=str, default=None, help='acoustics norm')
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_embedding', type=str, default=None, help='pretrained embedding')
	parser.add_argument('--use_type', type=str, default='True', help='use char level prediction')

	# model
	parser.add_argument('--acous_dim', type=int, default=24, help='acoustic feature dimension')
	parser.add_argument('--acous_norm', type=str, default='False', help='input acoustic fbk normalisation')
	parser.add_argument('--spec_aug', type=str, default='False', help='spectrum augmentation')
	parser.add_argument('--batch_norm', type=str, default='False', help='layer batch normalisation')
	parser.add_argument('--enc_mode', type=str, default='pyramid',
		help='acoustic lstm encoder structure - pyramid | cnn')

	parser.add_argument('--embedding_size', type=int, default=200, help='embedding size')
	parser.add_argument('--acous_hidden_size', type=int, default=200, help='acoustics hidden size')
	parser.add_argument('--acous_att_mode', type=str, default='bahdanau',
		help='attention mechanism mode - bahdanau / hybrid / dot_prod')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--hidden_size_shared', type=int, default=200,
		help='transformed att output hidden size (set as hidden_size_enc)')
	parser.add_argument('--num_unilstm_dec', type=int, default=2, help='number of encoder bilstm layers')

	# data
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--batch_first', type=str, default='True', help='batch as the first dimension')

	# train
	parser.add_argument('--random_seed', type=int, default=333, help='random seed')
	parser.add_argument('--acous_max_len', type=int, default=1500, help='maximum acoustic sequence length')
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--minibatch_partition', type=int, default=20, help='separate into minibatch - avoid OOM')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epoches')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--normalise_loss', type=str, default='True', help='normalise loss or not')
	parser.add_argument('--residual', type=str, default='True', help='residual connection')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='ratio of teacher forcing')
	parser.add_argument('--scheduled_sampling', type=str, default='False',
		help='gradually turn off teacher forcing')
	parser.add_argument('--max_grad_norm', type=float, default=1.0,
		help='optimiser gradient norm clipping: max grad norm')

	# save and print
	parser.add_argument('--max_count_no_improve', type=int, default=2,
		help='if meet max, operate roll back')
	parser.add_argument('--max_count_num_rollback', type=int, default=2,
		help='if meet max, reduce learning rate')
	parser.add_argument('--keep_num', type=int, default=1,
		help='number of models to keep')
	parser.add_argument('--checkpoint_every', type=int, default=10,
		help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10,
		help='print every n steps')

	return parser


def main():

	# import pdb; pdb.set_trace()
	# load config
	parser = argparse.ArgumentParser(description='LAS Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# set random seed
	if config['random_seed'] is not None:
		set_global_seeds(config['random_seed'])

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# resume or not
	if type(config['load']) != type(None):
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					batch_size=config['batch_size'],
					minibatch_partition=config['minibatch_partition'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					learning_rate=config['learning_rate'],
					eval_with_mask=config['eval_with_mask'],
					scheduled_sampling=config['scheduled_sampling'],
					teacher_forcing_ratio=config['teacher_forcing_ratio'],
					use_gpu=config['use_gpu'],
					max_grad_norm=config['max_grad_norm'],
					max_count_no_improve=config['max_count_no_improve'],
					max_count_num_rollback=config['max_count_num_rollback'],
					keep_num=config['keep_num'],
					normalise_loss=config['normalise_loss'])

	# vocab
	path_vocab_src = config['path_vocab_src']

	# load train set
	train_path_src = config['train_path_src']
	train_acous_path = config['train_acous_path']
	train_set = Dataset(train_path_src, path_vocab_src=path_vocab_src,
		use_type=config['use_type'],
		acous_path=train_acous_path,
		seqrev=config['seqrev'],
		acous_norm=config['acous_norm'],
		acous_norm_path=config['acous_norm_path'],
		max_seq_len=config['max_seq_len'],
		batch_size=config['batch_size'],
		acous_max_len=config['acous_max_len'],
		use_gpu=config['use_gpu'],
		logger=t.logger)

	vocab_size = len(train_set.vocab_src)

	# load dev set
	if config['dev_path_src']:
		dev_path_src = config['dev_path_src']
		dev_acous_path = config['dev_acous_path']
		dev_set = Dataset(dev_path_src, path_vocab_src=path_vocab_src,
			use_type=config['use_type'],
			acous_path=dev_acous_path,
			acous_norm_path=config['acous_norm_path'],
			seqrev=config['seqrev'],
			acous_norm=config['acous_norm'],
			max_seq_len=config['max_seq_len'],
			batch_size=config['batch_size'],
			acous_max_len=config['acous_max_len'],
			use_gpu=config['use_gpu'],
			logger=t.logger)
	else:
		dev_set = None

	# construct model
	las_model = LAS(vocab_size,
					embedding_size=config['embedding_size'],
					acous_hidden_size=config['acous_hidden_size'],
					acous_att_mode=config['acous_att_mode'],
					hidden_size_dec=config['hidden_size_dec'],
					hidden_size_shared=config['hidden_size_shared'],
					num_unilstm_dec=config['num_unilstm_dec'],
					#
					acous_dim=config['acous_dim'],
					acous_norm=config['acous_norm'],
					spec_aug=config['spec_aug'],
					batch_norm=config['batch_norm'],
					enc_mode=config['enc_mode'],
					use_type=config['use_type'],
					#
					embedding_dropout=config['embedding_dropout'],
					dropout=config['dropout'],
					residual=config['residual'],
					batch_first=config['batch_first'],
					max_seq_len=config['max_seq_len'],
					load_embedding=config['load_embedding'],
					word2id=train_set.src_word2id,
					id2word=train_set.src_id2word,
					use_gpu=config['use_gpu'])

	device = check_device(config['use_gpu'])
	t.logger.info('device:{}'.format(device))
	las_model = las_model.to(device=device)

	# run training
	las_model = t.train(
		train_set, las_model, num_epochs=config['num_epochs'], dev_set=dev_set)


if __name__ == '__main__':
	main()
