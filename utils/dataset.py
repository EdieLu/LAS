# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import torch
import collections
import codecs
import numpy as np
import random
from bpemb import BPEmb
from os.path import join

from utils.config import PAD, UNK, BOS, EOS, SPC


class IterDataset(torch.utils.data.Dataset):

	"""
		load features from

		'src_word_ids':train_src_word_ids[i_start:i_end],
		'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
		'acous_flis':train_acous_flis[i_start:i_end],
		'acous_spkids':train_acous_spkids[i_start:i_end],
		'acous_lengths':train_acous_lengths[i_start:i_end]
	"""

	def __init__(self, batches, acous_norm):

		super(Dataset).__init__()

		self.batches = batches
		self.acous_norm = acous_norm

	def __len__(self):

		return len(self.batches)

	def __getitem__(self, index):

		srcid = self.batches[index]['src_word_ids'] # lis
		srcid = torch.nn.utils.rnn.pad_sequence(
			[torch.LongTensor(elem) for elem in srcid], batch_first=True) # tensor
		srclen = self.batches[index]['src_sentence_lengths'] # lis
		acous_feat = self.load_file(index) # tensor
		acouslen = self.batches[index]['acous_lengths'] # lis

		return srcid, srclen, acous_feat, acouslen

	def load_file(self, index):

		# import pdb; pdb.set_trace()
		if self.acous_norm:
			norm_param = self.load_mu_std(index)
		else:
			norm_param = None
		acous_feat = self.load_acous_from_flis(index, norm_param=norm_param)

		return acous_feat

	def load_mu_std(self, index):

		spkids = self.batches[index]['acous_spkids']
		norm_param = []
		mydict = {}
		# base = '/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe/swbd-asr/prep/fbk_mu_std'
		base = '/home/alta/BLTSpeaking/exp-ytl28/encdec/lib-bpe/swbd-asr/prep/eval2000/fbk_mu_std'
			#only used for las evaluation
		# base = '/home/alta/BLTSpeaking/exp-ytl28/encdec/lib/add-acoustics/eval3/fbk_mu_std'
			#for eval3

		for idx in range(len(spkids)):
			spkid = spkids[idx]
			if spkid in mydict:
				pass
			else:
				f_mu = join(base, spkid+'.mu.npy')
				f_std = join(base, spkid+'.std.npy')
				mu = np.load(f_mu)
				std = np.load(f_std)
				mydict[spkid] = [mu, std]

			norm_param.append(mydict[spkid])

		return norm_param

	def load_acous_from_flis(self, index, norm_param=None):

		flis = self.batches[index]['acous_flis']
		max_len = 0
		feat_lis = []
		for idx in range(len(flis)):
			f = flis[idx]
			featarr = np.load(f)
			if type(norm_param) != type(None):
				mu, std = norm_param[idx] # dim=40
				featarr = 1. * (featarr - mu) / std
			feat = torch.FloatTensor(featarr) # np array (len x 40)
			max_len = max(max_len, feat.size(0))
			feat_lis.append(feat)

		# import pdb; pdb.set_trace()
		divisible_eight = max_len + 8 - max_len % 8
		dummy = torch.ones(divisible_eight , 40)
		feat_lis.append(dummy)
		feat_lis = torch.nn.utils.rnn.pad_sequence(feat_lis, batch_first=True)[:-1]

		return feat_lis


class Dataset(object):

	""" load src-tgt from file """

	def __init__(self,
		# add params
		path_src,
		path_tgt,
		path_vocab_src,
		path_vocab_tgt,
		acous_path=None,
		max_seq_len=32,
		batch_size=64,
		use_gpu=True,
		tsv_path=None,
		seqrev=False,
		acous_norm=False,
		use_type='char'
		):

		super(Dataset, self).__init__()

		self.path_src = path_src
		self.path_tgt = path_tgt
		self.acous_path = acous_path
		self.path_vocab_src = path_vocab_src
		self.path_vocab_tgt = path_vocab_tgt
		self.max_seq_len = max_seq_len
		self.batch_size = batch_size
		self.use_gpu = use_gpu

		self.tsv_path = tsv_path
		self.seqrev = seqrev
		self.acous_norm=acous_norm
		self.use_type = use_type

		self.acous_max_len = 6200
		# 3000 for train  | 6200 for eval3 should include all acous
		print('acous path: {}'.format(self.acous_path))
		print('max acous length: {}'.format(self.acous_max_len))

		self.load_vocab()
		self.load_sentences()
		self.load_acous_flis()
		self.preprocess()


	def load_vocab(self):

		# import pdb; pdb.set_trace()
		self.vocab_src = []
		with codecs.open(self.path_vocab_src, encoding='UTF-8') as f:
			vocab_src_lines	= f.readlines()

		self.src_word2id = collections.OrderedDict()
		self.src_id2word = collections.OrderedDict()

		for i, word in enumerate(vocab_src_lines):
			if word == '\n':
				continue
			word = word.strip().split()[0] # remove \n
			self.vocab_src.append(word)
			self.src_word2id[word] = i
			self.src_id2word[i] = word

		if type(self.path_vocab_tgt) != type(None):
			self.vocab_tgt = []
			with codecs.open(self.path_vocab_tgt, encoding='UTF-8') as f:
				vocab_tgt_lines = f.readlines()

			self.tgt_word2id = collections.OrderedDict()
			self.tgt_id2word = collections.OrderedDict()
			for i, word in enumerate(vocab_tgt_lines):
				word = word.strip().split()[0] # remove \n
				self.vocab_tgt.append(word)
				self.tgt_word2id[word] = i
				self.tgt_id2word[i] = word


	def load_sentences(self):

		with codecs.open(self.path_src, encoding='UTF-8') as f:
			self.src_sentences = f.readlines()

		if type(self.path_vocab_tgt) != type(None):
			with codecs.open(self.path_tgt, encoding='UTF-8') as f:
				self.tgt_sentences = f.readlines()

			assert len(self.src_sentences) == len(self.tgt_sentences), \
				'Mismatch src:tgt - {}:{}' \
				.format(len(self.src_sentences),len(self.tgt_sentences))

		if self.seqrev:
			for idx in range(len(self.src_sentences)):
				src_sent_rev = self.src_sentences[idx].strip().split()[::-1]
				tgt_sent_rev = self.tgt_sentences[idx].strip().split()[::-1]
				self.src_sentences[idx] = ' '.join(src_sent_rev)
				self.tgt_sentences[idx] = ' '.join(tgt_sent_rev)


	def load_tsv(self):

		""" laod the probability of each src word being disfluency or filler """

		if self.tsv_path == None:
			self.ddfd_seq_labs = None
		else:
			with codecs.open(self.tsv_path, encoding='UTF-8') as f:
				lines = f.readlines()

				lab_seq = []
				self.ddfd_seq_labs = []
				for line in lines:
					if line == '\n':
						if len(lab_seq):
							if self.seqrev:
								# reverse sequence for reverse decoding
								lab_seq = lab_seq[::-1]
							self.ddfd_seq_labs.append(lab_seq)
							lab_seq = []
					else:
						elems = line.strip().split('\t')
						tok = elems[0]
						if len(elems) == 3:
							lab = elems[-1][-1]
							if self.keep_filler:
								pass
							else:
								fl = elems[-2]
								if fl != '-':
									lab = 'E'
						elif len(elems) == 2:
							lab = elems[-1]
						else:
							assert False, \
								'check tsv file, requires either 2 or 3 elems per line'

						lab_seq.append(lab)

		assert len(self.src_sentences)==len(self.ddfd_seq_labs), \
			'Mismatch src:ddfd_lab - {}:{}'.format(
			len(self.src_sentences),len(self.ddfd_seq_labs))


	def load_acous_flis(self):

		""" load acoustic npy file list """

		self.acous_flis = []
		self.acous_length_lis = []
		self.acous_spkids = []
		if type(self.acous_path) == type(None):
			pass
		else:
			f = open(self.acous_path, 'r')
			lines = f.readlines()
			for line in lines:
				elems = line.strip().split()
				fname = elems[0]
				length = int(elems[1])
				spkid = elems[2].split('.')[0] # sw04004A
				self.acous_flis.append(fname)
				self.acous_length_lis.append(length)
				self.acous_spkids.append(spkid)


	def preprocess(self):

		"""
			used for LAS specifically

			Use:
				map word2id once for all epoches (improved data loading efficiency)
				shuffling is done later
			Returns:
				0 - over the entire epoch
				1 - ids of src
				src:	a  cat cat sat on the mat EOS PAD PAD ...
			Note:
				split into words
			Create
				self.train_src_word_ids
				self.train_src_sentence_lengths
				self.train_acous_flis
		"""

		self.vocab_size = {'src': len(self.src_word2id)}
		print("num_vocab: ", self.vocab_size['src'])

		# declare temporary vars
		train_src_word_ids = []
		train_src_sentence_lengths = []
		train_acous_flis = []
		train_acous_spkids = []
		train_acous_lengths = []

		assert len(self.acous_flis) == len(self.src_sentences), \
			'mismatch acoustics and sentences'

		for idx in range(len(self.src_sentences)):
			# import pdb; pdb.set_trace()
			src_sentence = self.src_sentences[idx]
			if self.use_type == 'char':
				src_words = src_sentence.strip()
			elif self.use_type == 'word':
				src_words = src_sentence.strip().split()
			elif self.use_type == 'bpe':
				# already converted to bpe
				src_words = src_sentence.strip().split()

			# ignore long seq of words
			if len(src_words) > self.max_seq_len - 1:
				# src + EOS
				continue

			# emtry seq - caused by [vocalised-noise]
			if len(src_words) == 0:
				continue

			# ignore long seq of acoustic features
			if self.acous_length_lis[idx] > self.acous_max_len:
				continue
			else:
				train_acous_flis.append(self.acous_flis[idx])
				train_acous_spkids.append(self.acous_spkids[idx])
				train_acous_lengths.append(self.acous_length_lis[idx])

			# source
			src_ids = []
			for i, word in enumerate(src_words):
				if word == ' ':
					assert self.use_type == 'char'
					src_ids.append(SPC)
				elif word in self.src_word2id:
					src_ids.append(self.src_word2id[word])
				else:
					src_ids.append(UNK)
			src_ids.append(EOS)
			assert src_ids[0] != PAD

			train_src_word_ids.append(src_ids)
			train_src_sentence_lengths.append(len(src_words)+1) # include one EOS

		# import pdb; pdb.set_trace()
		assert (len(train_src_word_ids) == len(train_acous_flis)), \
			"train_src_word_ids != train_acous_flis"

		self.num_training_sentences = len(train_src_word_ids)
		print("num_sentences: ", self.num_training_sentences)
			# only those that are not too long

		# set class var to be used in batchify
		self.train_src_word_ids = train_src_word_ids
		self.train_src_sentence_lengths = train_src_sentence_lengths
		self.train_acous_flis = train_acous_flis # list of acous npy fnames
		self.train_acous_spkids = train_acous_spkids
		self.train_acous_lengths = train_acous_lengths


	def construct_batches(self, is_train=False):

		"""
			Args:
				is_train: switch on shuffling is is_train
			Returns:
				batches of dataset
				src:
				a  SPC c a t SPC s a t SPC o n SPC t h e SPC m a t EOS PAD PAD ...
		"""

		# organise by length
		_x = list(zip(self.train_src_word_ids, self.train_src_sentence_lengths,
			self.train_acous_flis, self.train_acous_spkids, self.train_acous_lengths))
		if is_train:
			_x = sorted(_x, key=lambda l:l[1])
			# random.shuffle(_x)
		train_src_word_ids, train_src_sentence_lengths, train_acous_flis, \
			train_acous_spkids, train_acous_lengths = zip(*_x)

		# manual batching to allow shuffling by pt dataloader
		n_batches = int(self.num_training_sentences/self.batch_size +
			(self.num_training_sentences % self.batch_size > 0))
		batches = []
		for i in range(n_batches):
			i_start = i * self.batch_size
			i_end = min(i_start + self.batch_size, self.num_training_sentences)
			batch = {
				'src_word_ids':train_src_word_ids[i_start:i_end],
				'src_sentence_lengths': train_src_sentence_lengths[i_start:i_end],
				'acous_flis':train_acous_flis[i_start:i_end],
				'acous_spkids':train_acous_spkids[i_start:i_end],
				'acous_lengths':train_acous_lengths[i_start:i_end]
			}
			batches.append(batch)

		# pt dataloader
		params = {'batch_size': 1,
					'shuffle': is_train,
					'num_workers': 0}

		self.iter_set = IterDataset(batches, self.acous_norm)
		self.iter_loader = torch.utils.data.DataLoader(self.iter_set, **params)
		# import pdb; pdb.set_trace()


	def my_collate(self, batch):

		""" srcid, srclen, acous_feat, acouslen """

		srcid = [torch.LongTensor(item[0]) for item in batch]
		srclen = [item[1] for item in batch]
		acous_feat = [torch.Tensor(item[2]) for item in batch]
		acouslen = [item[3] for item in batch]

		srcid = torch.nn.utils.rnn.pad_sequence(srcid, batch_first=True) # b x l
		acous_feat = torch.nn.utils.rnn.pad_sequence(acous_feat, batch_first=True)
					# b x l x 40

		return [srcid, srclen, acous_feat, acouslen]


def load_pretrained_embedding(word2id, embedding_matrix, embedding_path):

	""" assign value to src_word_embeddings and tgt_word_embeddings """

	counter = 0
	with codecs.open(embedding_path, encoding="UTF-8") as f:
		for line in f:
			items = line.strip().split()
			if len(items) <= 2:
				continue
			word = items[0].lower()
			if word in word2id:
				id = word2id[word]
				vector = np.array(items[1:])
				embedding_matrix[id] = vector
				counter += 1

	print('loaded pre-trained embedding:', embedding_path)
	print('embedding vectors found:', counter)

	return embedding_matrix


def load_pretrained_embedding_bpe(embedding_matrix):

	""" load bpe embedding; add <pad> as id=0 """

	bpemb = BPEmb(lang="en", vs=25000, dim=200)
	embedding_matrix[1:] = bpemb.vectors
	print('loaded bpe pre-trained embedding')
	print('embedding vectors count:', embedding_matrix.shape[0])

	return embedding_matrix
