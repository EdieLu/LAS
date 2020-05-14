import torch
import torch.utils.tensorboard
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

sys.path.append('/home/alta/BLTSpeaking/exp-ytl28/local-ytl/acous-las-v3/')
from utils.misc import set_global_seeds, print_config, save_config, check_srctgt, check_src, check_src_print, check_src_tensor_print, validate_config, get_memory_alloc
from utils.misc import convert_dd_att_ref, load_acous_from_flis, load_mu_std
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor, _convert_to_tensor_pad, _del_var
from utils.dataset import Dataset
from utils.config import PAD, EOS
from modules.loss import NLLLoss, BCELoss, CrossEntropyLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import LAS

logging.basicConfig(level=logging.DEBUG)
device = torch.device('cpu')

MAX_COUNT_NO_IMPROVE = 5
MAX_COUNT_NUM_ROLLBACK = 5
KEEP_NUM = 2

def load_arguments(parser):

	# paths
	parser.add_argument('--train_path_src', type=str, required=True, help='train src dir')
	parser.add_argument('--train_path_tgt', type=str, default=None, help='train tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, default=None, help='vocab tgt dir')
	parser.add_argument('--dev_path_src', type=str, default=None, help='dev src dir')
	parser.add_argument('--dev_path_tgt', type=str, default=None, help='dev tgt dir')
	parser.add_argument('--save', type=str, required=True, help='model save dir')
	parser.add_argument('--load', type=str, default=None, help='model load dir')
	parser.add_argument('--load_embedding', type=str, default=None, help='pretrained embedding')
	parser.add_argument('--use_type', type=str, default='True', help='use char level prediction')

	parser.add_argument('--train_tsv_path', type=str, default=None, help='train set tsv file for seqtag')
	parser.add_argument('--dev_tsv_path', type=str, default=None, help='dev set tsv file for seqtag')
	parser.add_argument('--train_acous_path', type=str, default=None, help='train set acoustics')
	parser.add_argument('--dev_acous_path', type=str, default=None, help='dev set acoustics')

	# model
	parser.add_argument('--acous_norm', type=str, default='False', help='input acoustic fbk normalisation')
	parser.add_argument('--spec_aug', type=str, default='False', help='spectrum augmentation')
	parser.add_argument('--batch_norm', type=str, default='False', help='layer batch normalisation')
	parser.add_argument('--enc_mode', type=str, default='pyramid', help='acoustic lstm encoder structure - pyramid | cnn')

	parser.add_argument('--embedding_size', type=int, default=200, help='embedding size')
	parser.add_argument('--acous_hidden_size', type=int, default=200, help='acoustics hidden size')
	parser.add_argument('--acous_att_mode', type=str, default='bahdanau', help='attention mechanism mode - bahdanau / hybrid / dot_prod')
	parser.add_argument('--hidden_size_dec', type=int, default=200, help='encoder hidden size')
	parser.add_argument('--hidden_size_shared', type=int, default=200, help='transformed att output hidden size (set as hidden_size_enc)')
	parser.add_argument('--num_unilstm_dec', type=int, default=2, help='number of encoder bilstm layers')

	# train
	parser.add_argument('--random_seed', type=int, default=666, help='random seed')
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--embedding_dropout', type=float, default=0.0, help='embedding dropout')
	parser.add_argument('--dropout', type=float, default=0.0, help='dropout')
	parser.add_argument('--num_epochs', type=int, default=10, help='number of training epoches')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--residual', type=str, default='True', help='residual connection')
	parser.add_argument('--max_grad_norm', type=float, default=1.0, help='optimiser gradient norm clipping: max grad norm')
	parser.add_argument('--batch_first', type=str, default='True', help='batch as the first dimension')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_with_mask', type=str, default='True', help='calc loss excluding padded words')
	parser.add_argument('--teacher_forcing_ratio', type=float, default=1.0, help='ratio of teacher forcing')
	parser.add_argument('--scheduled_sampling', type=str, default='False', help='gradually turn off teacher forcing (if True, use teacher_forcing_ratio as the starting point)')
	parser.add_argument('--seqrev', type=str, default='False', help='reverse src, tgt sequence')

	# save and print
	parser.add_argument('--checkpoint_every', type=int, default=10, help='save ckpt every n steps')
	parser.add_argument('--print_every', type=int, default=10, help='print every n steps')
	parser.add_argument('--minibatch_partition', type=int, default=20, help='separate into minibatch for each increment of minibatch_partition')

	return parser


class Trainer(object):

	def __init__(self, expt_dir='experiment',
		load_dir=None,
		checkpoint_every=100,
		print_every=100,
		minibatch_partition=20,
		use_gpu=False,
		learning_rate=0.001,
		max_grad_norm=1.0,
		eval_with_mask=True,
		scheduled_sampling=False,
		teacher_forcing_ratio=1.0,
		):

		self.optimizer = None
		self.checkpoint_every = checkpoint_every
		self.print_every = print_every
		self.minibatch_partition = minibatch_partition
		self.use_gpu = use_gpu
		self.learning_rate = learning_rate
		self.max_grad_norm = max_grad_norm
		self.eval_with_mask = eval_with_mask
		self.scheduled_sampling = scheduled_sampling
		self.teacher_forcing_ratio = teacher_forcing_ratio

		if not os.path.isabs(expt_dir):
			expt_dir = os.path.join(os.getcwd(), expt_dir)
		self.expt_dir = expt_dir
		if not os.path.exists(self.expt_dir):
			os.makedirs(self.expt_dir)
		self.load_dir = load_dir

		self.logger = logging.getLogger(__name__)
		self.writer = torch.utils.tensorboard.writer.SummaryWriter(log_dir=self.expt_dir)


	def _evaluate_batches(self, model, dataset):

		model.eval()

		las_match = 0
		las_total = 0
		las_resloss = 0
		las_resloss_norm = 0

		evaliter = iter(dataset.iter_loader)

		out_count = 0
		with torch.no_grad():
			for idx in range(len(evaliter)):
				batch_items = evaliter.next()
				# print(idx,len(evaliter) )
				# if idx > 5:
				# 	break
				# if idx < len(evaliter)-3:
				# 	continue

				# print(idx)
				# import pdb; pdb.set_trace()
				# index [0]: since batch_size = 1 in dataloader
				batch_src_ids = batch_items[0][0]
				batch_src_lengths = batch_items[1]
				batch_acous_feats = batch_items[2][0]
				batch_acous_lengths = batch_items[3]

				batch_size = batch_src_ids.size(0)
				batch_seq_len = int(max(batch_src_lengths))
				batch_acous_len = int(max(batch_acous_lengths))

				n_minibatch_1 = int(batch_seq_len / self.minibatch_partition) + (batch_seq_len % self.minibatch_partition > 0)
				n_minibatch_2 = int(batch_acous_len / (self.minibatch_partition * 20)) + (batch_acous_len % (self.minibatch_partition * 20) > 0)
				n_minibatch = max(n_minibatch_1, n_minibatch_2)
				minibatch_size = int(batch_size / n_minibatch)
				n_minibatch += int((batch_size % minibatch_size > 0))

				# minibatch
				for bidx in range(n_minibatch):

					las_loss = NLLLoss()
					las_loss.reset()

					i_start = bidx * minibatch_size
					i_end = min(i_start + minibatch_size, batch_size)
					src_ids = batch_src_ids[i_start:i_end]
					src_lengths = batch_src_lengths[i_start:i_end]
					acous_feats = batch_acous_feats[i_start:i_end]
					acous_lengths = batch_acous_lengths[i_start:i_end]

					seq_len = max(src_lengths)
					acous_len = max(acous_lengths)
					acous_len = acous_len + 8 - acous_len % 8
					src_ids = src_ids[:,:seq_len].to(device=device)
					acous_feats = acous_feats[:,:acous_len].to(device=device)

					non_padding_mask_src = src_ids.data.ne(PAD)
					decoder_outputs, decoder_hidden, ret_dict = model(acous_feats, src_ids, is_training=False, use_gpu=self.use_gpu)

					# Evaluation
					logps = torch.stack(decoder_outputs, dim=1).to(device=device)
					las_loss.eval_batch_with_mask(logps.reshape(-1, logps.size(-1)), src_ids.reshape(-1), non_padding_mask_src.reshape(-1))
					las_loss.norm_term = torch.sum(non_padding_mask_src)
					las_resloss += las_loss.get_loss()
					las_resloss_norm += 1

					# las accuracy
					seqlist = ret_dict['sequence']
					seqres = torch.stack(seqlist, dim=1).to(device=device)
					correct = seqres.view(-1).eq(src_ids.reshape(-1)).masked_select(non_padding_mask_src.reshape(-1)).sum().item()
					las_match += correct
					las_total += non_padding_mask_src.sum().item()

					if out_count < 3 and idx > len(evaliter) - 5:
						srcwords = _convert_to_words_batchfirst(src_ids, dataset.src_id2word)
						seqwords = _convert_to_words(seqlist, dataset.src_id2word)
						outsrc = 'SRC: {}\n'.format(' '.join(srcwords[0])).encode('utf-8')
						outline = 'GEN: {}\n'.format(' '.join(seqwords[0])).encode('utf-8')
						sys.stdout.buffer.write(outsrc)
						sys.stdout.buffer.write(outline)
						out_count += 1
					torch.cuda.empty_cache()

		if las_total == 0:
			las_acc = float('nan')
		else:
			las_acc = las_match / las_total

		las_resloss /= las_resloss_norm
		accs = {'las_acc': las_acc}
		losses = {'las_loss': las_resloss}

		return accs, losses


	def _train_batch(self, model, batch_items, dataset, step, total_steps, src_labs=None):

		# -- DEBUG --
		# import pdb; pdb.set_trace()
		# print(step)
		# if step == 13:
		# 	import pdb; pdb.set_trace()

		# -- scheduled sampling --
		if not self.scheduled_sampling:
			teacher_forcing_ratio = self.teacher_forcing_ratio
		else:
			progress = 1.0 * step / total_steps
			teacher_forcing_ratio = 1.0 - progress

		# -- LOAD BATCH --
		batch_src_ids = batch_items[0][0]
		batch_src_lengths = batch_items[1]
		batch_acous_feats = batch_items[2][0]
		batch_acous_lengths = batch_items[3]

		# -- CONSTRUCT MINIBATCH --
		batch_size = batch_src_ids.size(0)
		batch_seq_len = int(max(batch_src_lengths))
		batch_acous_len = int(max(batch_acous_lengths))

		n_minibatch_1 = int(batch_seq_len / self.minibatch_partition) + (batch_seq_len % self.minibatch_partition > 0)
		n_minibatch_2 = int(batch_acous_len / (self.minibatch_partition * 20)) + (batch_acous_len % (self.minibatch_partition * 20) > 0)
		n_minibatch = max(n_minibatch_1, n_minibatch_2)
		minibatch_size = int(batch_size / n_minibatch)
		n_minibatch += int((batch_size % minibatch_size > 0))
		las_resloss = 0

		# minibatch
		for bidx in range(n_minibatch):

			# define loss
			las_loss = NLLLoss()
			las_loss.reset()

			# load data
			i_start = bidx * minibatch_size
			i_end = min(i_start + minibatch_size, batch_size)
			src_ids = batch_src_ids[i_start:i_end]
			src_lengths = batch_src_lengths[i_start:i_end]
			acous_feats = batch_acous_feats[i_start:i_end]
			acous_lengths = batch_acous_lengths[i_start:i_end]

			seq_len = max(src_lengths)
			acous_len = max(acous_lengths)
			acous_len = acous_len + 8 - acous_len % 8
			src_ids = src_ids[:,:seq_len].to(device=device)
			acous_feats = acous_feats[:,:acous_len].to(device=device)

			# sanity check src
			if step == 1: check_src_tensor_print(src_ids, dataset.src_id2word)

			# get padding mask
			non_padding_mask_src = src_ids.data.ne(PAD)

			# Forward propagation
			decoder_outputs, decoder_hidden, ret_dict = model(acous_feats, src_ids, is_training=True, teacher_forcing_ratio=teacher_forcing_ratio, use_gpu=self.use_gpu)

			logps = torch.stack(decoder_outputs, dim=1).to(device=device)
			las_loss.eval_batch_with_mask(logps.reshape(-1, logps.size(-1)), src_ids.reshape(-1), non_padding_mask_src.reshape(-1))
			las_loss.norm_term = 1.0 * torch.sum(non_padding_mask_src)

			# import pdb; pdb.set_trace()
			# Backward propagation: accumulate gradient
			las_loss.acc_loss /= n_minibatch
			las_loss.backward()
			las_resloss += las_loss.get_loss()
			torch.cuda.empty_cache()

		# import pdb; pdb.set_trace()
		self.optimizer.step()
		model.zero_grad()
		losses = {'las_loss': las_resloss}

		return losses


	def _train_epoches(self, train_set, model, n_epochs, start_epoch, start_step, dev_set=None):

		log = self.logger

		las_print_loss_total = 0  # Reset every print_every

		step = start_step
		step_elapsed = 0
		prev_acc = 0.0
		count_no_improve = 0
		count_num_rollback = 0
		ckpt = None

		# ******************** [loop over epochs] ********************
		for epoch in range(start_epoch, n_epochs + 1):

			for param_group in self.optimizer.optimizer.param_groups:
				print('epoch:{} lr: {}'.format(epoch, param_group['lr']))
				lr_curr = param_group['lr']

			# ----------construct batches-----------
			print('--- construct train set ---')
			train_set.construct_batches(is_train=True)
			if dev_set is not None:
				print('--- construct dev set ---')
				dev_set.construct_batches(is_train=True)

			# --------print info for each epoch----------
			steps_per_epoch = len(train_set.iter_loader)
			total_steps = steps_per_epoch * n_epochs
			log.info("steps_per_epoch {}".format(steps_per_epoch))
			log.info("total_steps {}".format(total_steps))

			log.debug(" ----------------- Epoch: %d, Step: %d -----------------" % (epoch, step))
			mem_kb, mem_mb, mem_gb = get_memory_alloc()
			mem_mb = round(mem_mb, 2)
			print('Memory used: {0:.2f} MB'.format(mem_mb))
			self.writer.add_scalar('Memory_MB', mem_mb, global_step=step)
			sys.stdout.flush()

			# ******************** [loop over batches] ********************
			model.train(True)
			trainiter = iter(train_set.iter_loader)
			for idx in range(steps_per_epoch):

				# load batch items
				batch_items = trainiter.next()

				# debug
				# if idx < steps_per_epoch - 2:
				# 	print(idx, steps_per_epoch)
				# 	continue
				# import pdb; pdb.set_trace()

				# update macro count
				step += 1
				step_elapsed += 1

				# Get loss
				losses = self._train_batch(model, batch_items, train_set, step, total_steps)

				las_loss = losses['las_loss']
				las_print_loss_total += las_loss

				if step % self.print_every == 0 and step_elapsed > self.print_every:
					las_print_loss_avg = las_print_loss_total / self.print_every
					las_print_loss_total = 0
					log_msg = 'Progress: %d%%, Train las: %.4f' % (step / total_steps * 100, las_print_loss_avg)
					log.info(log_msg)
					self.writer.add_scalar('train_las_loss', las_print_loss_avg, global_step=step)

				# Checkpoint
				if step % self.checkpoint_every == 0 or step == total_steps:

					# save criteria
					if dev_set is not None:
						dev_accs, dev_losses =  self._evaluate_batches(model, dev_set)
						las_loss = dev_losses['las_loss']
						las_acc = dev_accs['las_acc']
						log_msg = 'Progress: %d%%, Dev las loss: %.4f, accuracy: %.4f' % (step / total_steps * 100, las_loss, las_acc)
						log.info(log_msg)
						self.writer.add_scalar('dev_las_loss', las_loss, global_step=step)
						self.writer.add_scalar('dev_las_acc', las_acc, global_step=step)

						accuracy = las_acc
						# save
						if prev_acc < accuracy:
							# save best model
							ckpt = Checkpoint(model=model,
									   optimizer=self.optimizer,
									   epoch=epoch, step=step,
									   input_vocab=train_set.vocab_src,
									   output_vocab=train_set.vocab_src)

							saved_path = ckpt.save(self.expt_dir)
							print('saving at {} ... '.format(saved_path))
							# reset
							prev_acc = accuracy
							count_no_improve = 0
							count_num_rollback = 0
						else:
							count_no_improve += 1

						# roll back
						if count_no_improve > MAX_COUNT_NO_IMPROVE:
							# resuming
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								print('epoch:{} step: {} - rolling back {} ...'.format(epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)
								# start_epoch = resume_checkpoint.epoch
								# step = resume_checkpoint.step

							# reset
							count_no_improve = 0
							count_num_rollback += 1

						# update learning rate
						if count_num_rollback > MAX_COUNT_NUM_ROLLBACK:

							# roll back
							latest_checkpoint_path = Checkpoint.get_latest_checkpoint(self.expt_dir)
							if type(latest_checkpoint_path) != type(None):
								resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
								print('epoch:{} step: {} - rolling back {} ...'.format(epoch, step, latest_checkpoint_path))
								model = resume_checkpoint.model
								self.optimizer = resume_checkpoint.optimizer
								# A walk around to set optimizing parameters properly
								resume_optim = self.optimizer.optimizer
								defaults = resume_optim.param_groups[0]
								defaults.pop('params', None)
								defaults.pop('initial_lr', None)
								self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)
								start_epoch = resume_checkpoint.epoch
								step = resume_checkpoint.step

							# decrease lr
							for param_group in self.optimizer.optimizer.param_groups:
								param_group['lr'] *= 0.5
								lr_curr = param_group['lr']
								print('reducing lr ...')
								print('step:{} - lr: {}'.format(step, param_group['lr']))

							# check early stop
							if lr_curr < 0.000125:
								print('early stop ...')
								break

							# reset
							count_no_improve = 0
							count_num_rollback = 0

						model.train(mode=True)

					else:
						ckpt = Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_src)
						saved_path = ckpt.save(self.expt_dir)
						# saved_path = ckpt.save_epoch(self.expt_dir, epoch)
						print('saving at {} ... '.format(saved_path))

					if ckpt is None:
						ckpt = Checkpoint(model=model,
								   optimizer=self.optimizer,
								   epoch=epoch, step=step,
								   input_vocab=train_set.vocab_src,
								   output_vocab=train_set.vocab_src)
					ckpt.rm_old(self.expt_dir, keep_num=KEEP_NUM)
					print('n_no_improve {}, num_rollback {}'.format(count_no_improve, count_num_rollback))
				sys.stdout.flush()

			else:
				continue
			# break nested for loop
			break


	def train(self, train_set, model, num_epochs=5, optimizer=None, dev_set=None):

		"""
			Run training for a given model.
			Args:
				train_set: dataset
				dev_set: dataset, optional
				model: model to run training on, if `resume=True`, it would be
				   overwritten by the model loaded from the latest checkpoint.
				num_epochs (int, optional): number of epochs to run (default 5)
				optimizer (self.optim.Optimizer, optional): optimizer for training
				   (default: Optimizer(pytorch.optim.Adam, max_grad_norm=5))

			Returns:
				model (self.models): trained model.
		"""

		log = self.logger.info('MAX_COUNT_NO_IMPROVE: {}'.format(MAX_COUNT_NO_IMPROVE))
		log = self.logger.info('MAX_COUNT_NUM_ROLLBACK: {}'.format(MAX_COUNT_NUM_ROLLBACK))

		torch.cuda.empty_cache()
		if type(self.load_dir) != type(None):
			latest_checkpoint_path = self.load_dir
			print('resuming {} ...'.format(latest_checkpoint_path))
			resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
			model = resume_checkpoint.model
			print(model)
			self.optimizer = resume_checkpoint.optimizer

			# A walk around to set optimizing parameters properly
			resume_optim = self.optimizer.optimizer
			defaults = resume_optim.param_groups[0]
			defaults.pop('params', None)
			defaults.pop('initial_lr', None)
			self.optimizer.optimizer = resume_optim.__class__(model.parameters(), **defaults)

			model.set_idmap(train_set.src_word2id, train_set.src_id2word)
			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			# just for the sake of finetuning
			# start_epoch = 1
			# step = 0
			start_epoch = resume_checkpoint.epoch
			step = resume_checkpoint.step

		else:
			start_epoch = 1
			step = 0
			print(model)

			for name, param in model.named_parameters():
				log = self.logger.info('{}:{}'.format(name, param.size()))

			if optimizer is None:
				optimizer = Optimizer(torch.optim.Adam(model.parameters(),
							lr=self.learning_rate), max_grad_norm=self.max_grad_norm) # 5 -> 1
			self.optimizer = optimizer

		self.logger.info("Optimizer: %s, Scheduler: %s" % (self.optimizer.optimizer, self.optimizer.scheduler))

		self._train_epoches(train_set, model, num_epochs, start_epoch, step, dev_set=dev_set)

		return model


def main():

	# import pdb; pdb.set_trace()
	# load config
	parser = argparse.ArgumentParser(description='PyTorch LAS Training')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	set_global_seeds(config['random_seed'])

	# record config
	if not os.path.isabs(config['save']):
		config_save_dir = os.path.join(os.getcwd(), config['save'])
	if not os.path.exists(config['save']):
		os.makedirs(config['save'])

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# resume or not
	if type(config['load']) != type(None):
		config_save_dir = os.path.join(config['save'], 'model-cont.cfg')
	else:
		config_save_dir = os.path.join(config['save'], 'model.cfg')
	save_config(config, config_save_dir)

	# vocab
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']

	# load train set
	train_path_src = config['train_path_src']
	train_path_tgt = config['train_path_tgt']
	train_tsv_path = config['train_tsv_path']
	train_acous_path = config['train_acous_path']
	train_set = Dataset(train_path_src, train_path_tgt, path_vocab_src, path_vocab_tgt, use_type=config['use_type'],
						tsv_path=train_tsv_path, acous_path=train_acous_path, seqrev=config['seqrev'], acous_norm=config['acous_norm'],
						max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
						use_gpu=config['use_gpu'])
	# load dev set
	if config['dev_path_src']:
		dev_path_src = config['dev_path_src']
		dev_path_tgt = config['dev_path_tgt']
		dev_tsv_path = config['dev_tsv_path']
		dev_acous_path = config['dev_acous_path']
		dev_set = Dataset(dev_path_src, dev_path_tgt, path_vocab_src, path_vocab_tgt, use_type=config['use_type'],
						tsv_path=dev_tsv_path, acous_path=dev_acous_path, seqrev=config['seqrev'], acous_norm=config['acous_norm'],
						max_seq_len=config['max_seq_len'], batch_size=config['batch_size'],
						use_gpu=config['use_gpu'])
	else:
		dev_set = None

	vocab_size = len(train_set.vocab_src)
	# construct model
	las_model = LAS(vocab_size,
					embedding_size=config['embedding_size'],
					acous_hidden_size=config['acous_hidden_size'],
					acous_att_mode=config['acous_att_mode'],
					hidden_size_dec=config['hidden_size_dec'],
					hidden_size_shared=config['hidden_size_shared'],
					num_unilstm_dec=config['num_unilstm_dec'],
					#
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

	if config['use_gpu']: las_model = las_model.cuda()

	# contruct trainer
	t = Trainer(expt_dir=config['save'],
					load_dir=config['load'],
					checkpoint_every=config['checkpoint_every'],
					print_every=config['print_every'],
					minibatch_partition=config['minibatch_partition'],
					learning_rate=config['learning_rate'],
					eval_with_mask=config['eval_with_mask'],
					scheduled_sampling=config['scheduled_sampling'],
					teacher_forcing_ratio=config['teacher_forcing_ratio'],
					use_gpu=config['use_gpu'],
					max_grad_norm=config['max_grad_norm'])

	# run training
	las_model = t.train(train_set, las_model, num_epochs=config['num_epochs'], dev_set=dev_set)


if __name__ == '__main__':
	main()
