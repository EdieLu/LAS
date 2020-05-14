import torch
import random
import time
import os
import logging
import argparse
import sys
import numpy as np

sys.path.append('/home/alta/BLTSpeaking/exp-ytl28/local-ytl/acous-las-v3/')
from utils.dataset import Dataset
from utils.misc import set_global_seeds, print_config, save_config, validate_config, get_memory_alloc, load_acous_from_flis
from utils.misc import _convert_to_words_batchfirst, _convert_to_words, _convert_to_tensor, _convert_to_tensor_pad, plot_alignment, plot_attention
from utils.config import PAD, EOS
from modules.loss import NLLLoss
from modules.optim import Optimizer
from modules.checkpoint import Checkpoint
from models.recurrent import LAS

import logging
logging.basicConfig(level=logging.INFO)

device = torch.device('cpu')

def load_arguments(parser):

	""" Seq2Seq-DD eval """

	# paths
	parser.add_argument('--test_path_src', type=str, required=True, help='test src dir')
	parser.add_argument('--test_path_tgt', type=str, default=None, help='test tgt dir')
	parser.add_argument('--path_vocab_src', type=str, required=True, help='vocab src dir')
	parser.add_argument('--path_vocab_tgt', type=str, default=None, help='vocab tgt dir')	
	parser.add_argument('--use_type', type=str, default='char', help='use char | word level prediction')
	parser.add_argument('--acous_norm', type=str, default='False', help='input acoustic fbk normalisation')

	parser.add_argument('--load', type=str, required=True, help='model load dir')
	parser.add_argument('--test_path_out', type=str, required=True, help='test out dir')

	parser.add_argument('--test_tsv_path', type=str, default=None, help='test set tsv')
	parser.add_argument('--test_acous_path', type=str, default=None, help='test set acoustics')

	# others
	parser.add_argument('--max_seq_len', type=int, default=32, help='maximum sequence length')
	parser.add_argument('--batch_size', type=int, default=64, help='batch size')	
	parser.add_argument('--beam_width', type=int, default=0, help='beam width; set to 0 to disable beam search')	
	parser.add_argument('--use_gpu', type=str, default='False', help='whether or not using GPU')
	parser.add_argument('--eval_mode', type=int, default=2, help='which evaluation mode to use')
	parser.add_argument('--seqrev', type=str, default=False, help='whether or not to reverse sequence')

	return parser


def translate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, seqrev=False):

	""" 
		no reference tgt given - Run translation.
		Args:
			test_set: test dataset
				src, tgt using the same dir
			test_path_out: output dir
			load_dir: model dir
			use_gpu: on gpu/cpu
	"""

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model.to(device)
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	print('max seq len {}'.format(model.max_seq_len))
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

				decoder_outputs, decoder_hidden, other = model(acous_feats, src_ids, is_training=False, use_gpu=use_gpu, beam_width=beam_width)
				# decoder_outputs, decoder_hidden, other = model(acous_feats, src_ids, is_training=False, teacher_forcing_ratio=1.0, use_gpu=use_gpu, beam_width=beam_width)

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

				elif test_set.use_type == 'word' or test_set.use_type == 'bpe':
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


def debug_beam_search(test_set, load_dir, use_gpu, max_seq_len, beam_width):

	""" 
		with reference tgt given - debug beam search.
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu
		Returns:
			accuracy (excluding PAD tokens)
	"""

	# load model
	# latest_checkpoint_path = Checkpoint.get_latest_checkpoint(load_dir)
	# latest_checkpoint_path = Checkpoint.get_thirdlast_checkpoint(load_dir)
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))	
	print('Model laoded')

	# reset batch_size:
	model.reset_max_seq_len(max_seq_len)
	model.reset_use_gpu(use_gpu)	
	model.reset_batch_size(test_set.batch_size)	
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()

	# load test
	if type(test_set.attkey_path) == type(None):
		test_batches, vocab_size = test_set.construct_batches(is_train=False)
	else:
		test_batches, vocab_size = test_set.construct_batches_with_ddfd_prob(is_train=False)

	model.eval()
	match = 0
	total = 0
	with torch.no_grad():
		for batch in test_batches:

			src_ids = batch['src_word_ids']
			src_lengths = batch['src_sentence_lengths']
			tgt_ids = batch['tgt_word_ids']
			tgt_lengths = batch['tgt_sentence_lengths']
			src_probs = None
			if 'src_ddfd_probs' in batch:
				src_probs =  batch['src_ddfd_probs']
				src_probs = _convert_to_tensor(src_probs, use_gpu).unsqueeze(2)

			src_ids = _convert_to_tensor(src_ids, use_gpu)
			tgt_ids = _convert_to_tensor(tgt_ids, use_gpu)

			decoder_outputs, decoder_hidden, other = model(src_ids, tgt_ids, 
															is_training=False,
															att_key_feats=src_probs,
															beam_width=beam_width)

			# Evaluation
			seqlist = other['sequence'] # traverse over time not batch
			if beam_width > 1: 
				# print('dict:sequence')
				# print(len(seqlist))
				# print(seqlist[0].size())

				full_seqlist = other['topk_sequence']
				# print('dict:topk_sequence')
				# print(len(full_seqlist))
				# print((full_seqlist[0]).size())
				# input('...')
				seqlists = []
				for i in range(beam_width):
					seqlists.append([seq[:, i] for seq in full_seqlist])

				# print(decoder_outputs[0].size())
				# print('tgt id size {}'.format(tgt_ids.size()))
				# input('...')

				decoder_outputs = decoder_outputs[:-1]
				# print(len(decoder_outputs))

			for step, step_output in enumerate(decoder_outputs): # loop over time steps
				target = tgt_ids[:, step+1]
				non_padding = target.ne(PAD)
				# print('step', step)
				# print('target', target)
				# print('hyp', seqlist[step])
				# if beam_width > 1: 
				# 	print('full_seqlist', full_seqlist[step])
				# input('...')
				correct = seqlist[step].view(-1).eq(target).masked_select(non_padding).sum().item()
				match += correct
				total += non_padding.sum().item()

			# write to file
			refwords = _convert_to_words_batchfirst(tgt_ids[:,1:], test_set.tgt_id2word)			
			seqwords = _convert_to_words(seqlist, test_set.tgt_id2word)
			seqwords_list = []
			for i in range(beam_width):
				seqwords_list.append(_convert_to_words(seqlists[i], test_set.tgt_id2word))

			for i in range(len(seqwords)):
				outline_ref = ' '.join(refwords[i]) 
				print('REF', outline_ref)				
				outline_hyp = ' '.join(seqwords[i]) 
				# print(outline_hyp)
				outline_hyps = []
				for j in range(beam_width):
					outline_hyps.append(' '.join(seqwords_list[j][i]))
					print('{}th'.format(j), outline_hyps[-1])

				# skip padding sentences in batch (num_sent % batch_size != 0)
				# if src_lengths[i] == 0:
				# 	continue
				# words = []
				# for word in seqwords[i]:
				# 	if word == '<pad>':
				# 		continue
				# 	elif word == '</s>':
				# 		break
				# 	else:
				# 		words.append(word)
				# if len(words) == 0:
				# 	outline = ''
				# else:
				# 	outline = ' '.join(words)

				input('...')

			sys.stdout.flush()

		if total == 0:
			accuracy = float('nan')
		else:
			accuracy = match / total
		
	return accuracy


def acous_att_plot(test_set, load_dir, plot_path, use_gpu, max_seq_len, beam_width):

	""" 
		generate attention alignment plots
		Args:
			test_set: test dataset
			load_dir: model dir
			use_gpu: on gpu/cpu			
			max_seq_len
		Returns:
			
	"""
	# import pdb; pdb.set_trace()
	use_gpu = False

	# load model
	latest_checkpoint_path = load_dir
	resume_checkpoint = Checkpoint.load(latest_checkpoint_path)

	model = resume_checkpoint.model
	print('Model dir: {}'.format(latest_checkpoint_path))
	print('Model laoded')

	# reset batch_size:
	model.cpu()
	model.reset_max_seq_len(max_seq_len)
	print('max seq len {}'.format(model.max_seq_len))
	sys.stdout.flush()
		
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

			# decoder_outputs, decoder_hidden, ret_dict = model(acous_feats, src_ids, is_training=False, use_gpu=use_gpu, beam_width=beam_width)
			decoder_outputs, decoder_hidden, ret_dict = model(acous_feats, src_ids, is_training=False, teacher_forcing_ratio=1.0, use_gpu=use_gpu, beam_width=beam_width)
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
			# import pdb; pdb.set_trace()
			loc_eos_k = srcwords[i].index('</s>') + 1
			print('eos_k: {}'.format(loc_eos_k))
			# loc_eos_m = seqwords[i].index('</s>') + 1
			loc_eos_m = len(seqwords[i])
			print('eos_m: {}'.format(loc_eos_m))

			att_score_trim = attention[:loc_eos_m, :] #each row (each query) sum up to 1
			# print(att_score_trim)
			# print('\n')
			
			choice = input('Plot or not ? - y/n\n')
			if choice:
				if choice.lower()[0] == 'y':
					# import pdb; pdb.set_trace()
					print('plotting ...')
					plot_dir = os.path.join(plot_path, '{}.png'.format(count))
					src = srcwords[i][:loc_eos_m]
					gen = seqwords[i][:loc_eos_m]

					# x-axis: acous; y-axis: src
					plot_attention(att_score_trim.numpy(), plot_dir, gen, words_right=src) # no ref 				
					count += 1
					input('Press enter to continue ...')



def main():

	# load config
	parser = argparse.ArgumentParser(description='PyTorch LAS DD Evaluation')
	parser = load_arguments(parser)
	args = vars(parser.parse_args())
	config = validate_config(args)
	config_save_dir = os.path.join(config['load'], 'eval.cfg') 
	save_config(config, config_save_dir)

	# check device:
	if config['use_gpu'] and torch.cuda.is_available():
		global device
		device = torch.device('cuda')
	else:
		device = torch.device('cpu')
	print('device: {}'.format(device))

	# load src-tgt pair
	test_path_src = config['test_path_src']
	test_path_tgt = config['test_path_tgt']
	path_vocab_src = config['path_vocab_src']
	path_vocab_tgt = config['path_vocab_tgt']
	test_path_out = config['test_path_out']
	test_tsv_path = config['test_tsv_path']
	test_acous_path = config['test_acous_path']

	load_dir = config['load']
	max_seq_len = config['max_seq_len']
	batch_size = config['batch_size']
	beam_width = config['beam_width']
	use_gpu = config['use_gpu']
	seqrev = config['seqrev']
	print('reverse seq: {}'.format(seqrev))
	print('use gpu: {}'.format(use_gpu))

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)

	if not os.path.exists(test_path_out):
		os.makedirs(test_path_out)
	config_save_dir = os.path.join(test_path_out, 'eval.cfg') 
	save_config(config, config_save_dir)

	# set test mode: 3 = DEBUG; 4 = PLOT
	MODE = config['eval_mode']
	if MODE == 3 or MODE == 4 or MODE == 6:
		max_seq_len = 32
		batch_size = 1
		beam_width = 1
		use_gpu = False

	# load test_set
	test_set = Dataset(test_path_src, test_path_tgt, path_vocab_src, path_vocab_tgt, use_type=config['use_type'],
						tsv_path=test_tsv_path, acous_path=test_acous_path, seqrev=seqrev, acous_norm=config['acous_norm'],
						max_seq_len=max_seq_len, batch_size=batch_size, use_gpu=use_gpu)
	print('Testset loaded')
	sys.stdout.flush()

	# run eval
	if MODE == 2: 
		translate(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width, seqrev=seqrev)

	elif MODE == 5:
		# debug for beam search
		debug_beam_search(test_set, load_dir, use_gpu, max_seq_len, beam_width)

	elif MODE == 6:
		# plotting las attn
		acous_att_plot(test_set, load_dir, test_path_out, use_gpu, max_seq_len, beam_width)



if __name__ == '__main__':
	main()

