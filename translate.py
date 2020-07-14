import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

from utils.dataset import Dataset
from utils.misc import save_config, validate_config, check_device
from utils.misc import get_memory_alloc
from utils.misc import _convert_to_words_batchfirst, _convert_to_words
from utils.misc import _convert_to_tensor, _convert_to_tensor_pad
from utils.misc import plot_alignment, plot_attention
from utils.config import PAD, EOS
from modules.checkpoint import Checkpoint
from models.Las import LAS

logging.basicConfig(level=logging.INFO)


def load_arguments(parser):

	""" LAS eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--path_vocab_src', type=str, default='None', help='vocab src dir, no need')
	parser.add_argument('--use_type', type=str, default='char', help='use char | word level prediction')
	parser.add_argument('--acous_norm', type=str, default='False', help='input acoustic fbk normalisation')
	parser.add_argument('--acous_norm_path', type=str, default=None, help='acoustics norm')
	parser.add_argument('--test_acous_path', type=str, default=None, help='test set acoustics')

	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--seqrev', type=str, default=False, help='whether or not to reverse sequence')
	parser.add_argument('--teacher_forcing', type=str, default='True', help='decode under teacher forcing or not')

	return parser


def translate(test_set, model, test_path_out, use_gpu,
	max_seq_len, beam_width, device, teacher_forcing_ratio=0.0, seqrev=False):

	"""
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# reset batch_size:
	model.decoder.max_seq_len = max_seq_len
	print('max seq len {}'.format(model.decoder.max_seq_len))
	sys.stdout.flush()

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)

	print('num batches: {}'.format(len(evaliter)))
	with open(os.path.join(test_path_out, 'translate.txt'), 'w', encoding="utf8") as f:
		model.eval()
		with torch.no_grad():
			for idx in range(len(evaliter)):

				batch_items = evaliter.next()
				src_ids = batch_items[0][0].to(device=device)
				src_lengths = batch_items[1]
				acous_feats = batch_items[2][0].to(device=device)
				acous_lengths = batch_items[3]

				decoder_outputs, decoder_hidden, other = \
					model(acous_feats, acous_lens=acous_lengths,
						tgt=src_ids, is_training=False,
				 		teacher_forcing_ratio=teacher_forcing_ratio,
						use_gpu=use_gpu, beam_width=beam_width)

				# memory usage
				mem_kb, mem_mb, mem_gb = get_memory_alloc()
				mem_mb = round(mem_mb, 2)
				print('Memory used: {0:.2f} MB'.format(mem_mb))
				batch_size = src_ids.size(0)

				# write to file
				# import pdb; pdb.set_trace()
				srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
				seqlist = other['sequence']
				seqwords = _convert_to_words(seqlist, test_set.src_id2word)

				if test_set.use_type == 'char':
					for i in range(len(seqwords)):
						# skip padding sentences in batch (num_sent % batch_size != 0)
						if src_lengths[i] == 0:
							continue
						words = []
						for word in seqwords[i]:
							if word == '<pad>':
								continue
							elif word == '</s>':
								break
							elif word == '<spc>':
								words.append(' ')
							else:
								words.append(word)
						if len(words) == 0:
							outline = ''
						else:
							if seqrev:
								words = words[::-1]
							outline = ''.join(words)
						f.write('{}\n'.format(outline))

				elif test_set.use_type == 'word':
					for i in range(len(seqwords)):
						if src_lengths[i] == 0:
							continue
						words = []
						for word in seqwords[i]:
							if word == '<pad>':
								continue
							elif word == '</s>':
								break
							else:
								words.append(word)
						if len(words) == 0:
							outline = ''
						else:
							if seqrev:
								words = words[::-1]
							outline = ' '.join(words)
						f.write('{}\n'.format(outline))

				sys.stdout.flush()


def acous_att_plot(test_set, model, plot_path, use_gpu, max_seq_len,
	beam_width, teacher_forcing_ratio=0.0):

	"""
		generate attention alignment plots
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
			max_seq_len
		Returns:

	"""

	# reset batch_size:
	model.cpu()
	model.reset_max_seq_len(max_seq_len)
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()
	device = 'cpu'

	# load test
	test_set.construct_batches(is_train=False)
	evaliter = iter(test_set.iter_loader)

	# start eval
	count=0
	model.eval()
	with torch.no_grad():
		for idx in range(len(evaliter)):

			batch_items = evaliter.next()
			src_ids = batch_items[0][0].to(device=device)
			src_lengths = batch_items[1]
			acous_feats = batch_items[2][0].to(device=device)
			acous_lengths = batch_items[3]

			# decoder_outputs, decoder_hidden, ret_dict = model(
				# acous_feats, src_ids, is_training=False,
				# use_gpu=use_gpu, beam_width=beam_width)
			decoder_outputs, decoder_hidden, ret_dict = model(
				acous_feats, acous_lens=acous_lengths, tgt=src_ids, is_training=False,
				teacher_forcing_ratio=1.0, use_gpu=use_gpu, beam_width=beam_width)
			# attention: [32 x ?] (batch_size x src_len x acous_len(key_len))
			# default batch_size = 1
			i = 0
			attention = torch.cat(ret_dict['attention_score'],dim=1)[i]
			bsize = test_set.batch_size
			max_seq = test_set.max_seq_len
			vocab_size = len(test_set.src_word2id)

			# Print sentence by sentence
			seqlist = ret_dict['sequence']
			seqwords = _convert_to_words(seqlist, test_set.src_id2word)
			outline_gen = ' '.join(seqwords[i])
			srcwords = _convert_to_words_batchfirst(src_ids, test_set.src_id2word)
			outline_src = ' '.join(srcwords[i])
			print('SRC: {}'.format(outline_src))
			print('GEN: {}'.format(outline_gen))

			# plotting
			loc_eos_k = srcwords[i].index('</s>') + 1
			print('eos_k: {}'.format(loc_eos_k))
			loc_eos_m = len(seqwords[i])
			print('eos_m: {}'.format(loc_eos_m))

			att_score_trim = attention[:loc_eos_m, :] #each row (each query) sum up to 1

			choice = input('Plot or not ? - y/n\n')
			if choice:
				if choice.lower()[0] == 'y':
					# import pdb; pdb.set_trace()
					print('plotting ...')
					plot_dir = os.path.join(plot_path, '{}.png'.format(count))
					src = srcwords[i][:loc_eos_m]
					gen = seqwords[i][:loc_eos_m]

					# x-axis: acous; y-axis: src, no ref
					plot_attention(att_score_trim.numpy(), plot_dir, gen, words_right=src)
					count += 1
					input('Press enter to continue ...')



def main():

	# load config
	parser = argparse.ArgumentParser(description='PyTorch LAS DD Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)

	# load src-tgt pair
	test_path_src = config['test_path_src']
	path_vocab_src = config['path_vocab_src']
	test_path_out = config['test_path_out']
	test_acous_path = config['test_acous_path']
	acous_norm_path = config['acous_norm_path']

	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)
	config_save_dir = os.path.join(test_path_out, 'eval.cfg')
	save_config(config, config_save_dir)

	# set test mode
	MODE = config['eval_mode']
	if MODE == 2:
		max_seq_len = 32
		batch_size = 1
		beam_width = 1
		use_gpu = False

	# check device:
	device = check_device(use_gpu)
	print('device: {}'.format(device))

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)
	model = resume_checkpoint.model.to(device)
	vocab_src = resume_checkpoint.input_vocab
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# load test_set
	test_set = Dataset(test_path_src, vocab_src_list=vocab_src,
						use_type=config['use_type'],
						acous_path=test_acous_path,
						acous_norm_path=config['acous_norm_path'],
						seqrev=seqrev,
						acous_norm=config['acous_norm'],
						max_seq_len=max_seq_len,
						batch_size=batch_size,
						acous_max_len=6000,
						use_gpu=use_gpu)
	print('Test dir: {}'.format(test_path_src))
	print('Testset loaded')
	sys.stdout.flush()

	# teacher forcing: teacher_forcing_ratio=1.0
	# free running: teacher_forcing_ratio=0.0
	if config['teacher_forcing']:
		teacher_forcing_ratio = 1.0
	else:
		teacher_forcing_ratio = 0.0

	# run eval:
	if MODE == 1:
		translate(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width, device,
			teacher_forcing_ratio=teacher_forcing_ratio, seqrev=seqrev)

	elif MODE == 2:
		# plotting las attn
		acous_att_plot(test_set, model, test_path_out, use_gpu,
			max_seq_len, beam_width,
			teacher_forcing_ratio=teacher_forcing_ratio)



if __name__ == '__main__':
	main()
