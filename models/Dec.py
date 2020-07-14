import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from modules.attention import AttentionLayer
from utils.config import PAD, EOS, BOS
from utils.dataset import load_pretrained_embedding
from utils.misc import check_device

import warnings
warnings.filterwarnings("ignore")

KEY_ATTN_SCORE = 'attention_score'
KEY_ATTN_OUT = 'attention_out'
KEY_LENGTH = 'length'
KEY_SEQUENCE = 'sequence'

class Dec(nn.Module):

	""" listen attend spell model + dd tag """

	def __init__(self,
		# params
		vocab_size,
		embedding_size=200,
		acous_hidden_size=256,
		acous_att_mode='bahdanau',
		hidden_size_dec=200,
		hidden_size_shared=200,
		num_unilstm_dec=4,
		use_type='char',
		#
		embedding_dropout=0,
		dropout=0.0,
		residual=True,
		batch_first=True,
		max_seq_len=32,
		load_embedding=None,
		word2id=None,
		id2word=None,
		hard_att=False,
		use_gpu=False
		):

		super(Dec, self).__init__()
		device = check_device(use_gpu)

		# define model
		self.acous_hidden_size = acous_hidden_size
		self.acous_att_mode = acous_att_mode
		self.hidden_size_dec = hidden_size_dec
		self.hidden_size_shared = hidden_size_shared
		self.num_unilstm_dec = num_unilstm_dec

		# define var
		self.hard_att = hard_att
		self.residual = residual
		self.max_seq_len = max_seq_len
		self.use_type = use_type

		# use shared embedding + vocab
		self.vocab_size = vocab_size
		self.embedding_size = embedding_size
		self.load_embedding = load_embedding
		self.word2id = word2id
		self.id2word = id2word

		# define operations
		self.embedding_dropout = nn.Dropout(embedding_dropout)
		self.dropout = nn.Dropout(dropout)

		# ------- load embeddings --------
		if self.load_embedding:
			embedding_matrix = np.random.rand(self.vocab_size, self.embedding_size)
			embedding_matrix = load_pretrained_embedding(
				self.word2id,embedding_matrix,self.load_embedding)
			embedding_matrix = torch.FloatTensor(embedding_matrix)
			self.embedder = nn.Embedding.from_pretrained(embedding_matrix,
				freeze=False, sparse=False,padding_idx=PAD)
		else:
			self.embedder = nn.Embedding(self.vocab_size, self.embedding_size,
				sparse=False, padding_idx=PAD)

		# ------ define acous att --------
		dropout_acous_att = dropout
		self.acous_hidden_size_att = 0 # ignored with bilinear

		self.acous_key_size = self.acous_hidden_size * 2 	# acous feats
		self.acous_value_size = self.acous_hidden_size * 2 	# acous feats
		self.acous_query_size = self.hidden_size_dec 		# use dec(words) as query
		self.acous_att = AttentionLayer(self.acous_query_size, self.acous_key_size,
									value_size=self.acous_value_size,
									mode=self.acous_att_mode,
									dropout=dropout_acous_att,
									query_transform=False,
									output_transform=False,
									hidden_size=self.acous_hidden_size_att,
									use_gpu=use_gpu,
									hard_att=False)

		# ------ define acous out --------
		self.acous_ffn = nn.Linear(self.acous_hidden_size * 2 + self.hidden_size_dec,
									self.hidden_size_shared, bias=False)
		self.acous_out = nn.Linear(self.hidden_size_shared, self.vocab_size, bias=True)


		# ------ define acous dec -------
		# embedding_size_dec + self.hidden_size_shared [200+200]-> hidden_size_dec [200]
		if not self.residual:
			self.dec = torch.nn.LSTM(
				self.embedding_size+self.hidden_size_shared, self.hidden_size_dec,
				num_layers=self.num_unilstm_dec, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=False)
		else:
			self.dec = nn.Module()
			self.dec.add_module('l0', torch.nn.LSTM(
				self.embedding_size+self.hidden_size_shared, self.hidden_size_dec,
				num_layers=1, batch_first=batch_first,
				bias=True, dropout=dropout, bidirectional=False))

			for i in range(1, self.num_unilstm_dec):
				self.dec.add_module('l'+str(i), torch.nn.LSTM(self.hidden_size_dec,
					self.hidden_size_dec,num_layers=1, batch_first=batch_first,
					bias=True,dropout=dropout, bidirectional=False))


	def check_var(self, var_name, var_val_set=None):

		""" to make old models capatible with added classvar in later versions """

		if not hasattr(self, var_name):
			var_val = var_val_set if type(var_val_set) != type(None) else None
			setattr(self, var_name, var_val)


	def forward(self, acous_outputs, acous_lens=None, tgt=None,
		hidden=None, is_training=False, teacher_forcing_ratio=0.0,
		beam_width=1, use_gpu=False):

		"""
			Args:
				enc_outputs: [batch_size, acous_len / 8, self.acous_hidden_size * 2]
				tgt: list of word_ids 			[b x seq_len]
				hidden: initial hidden state
				is_training: whether in eval or train mode
				teacher_forcing_ratio: default at 1 - always teacher forcing
			Returns:
				decoder_outputs: list of step_output -
					log predicted_softmax [batch_size, 1, vocab_size_dec] * (T-1)
		"""

		# import pdb; pdb.set_trace()

		global device
		device = check_device(use_gpu)

		# 0. init var
		ret_dict = dict()
		ret_dict[KEY_ATTN_SCORE] = []

		decoder_outputs = []
		sequence_symbols = []
		batch_size = acous_outputs.size(0)

		if type(tgt) == type(None):
			tgt = torch.Tensor([BOS]).repeat(
				batch_size, self.max_seq_len).type(torch.LongTensor).to(device=device)

		max_seq_len = tgt.size(1)
		lengths = np.array([max_seq_len] * batch_size)

		# 1. convert id to embedding
		emb_tgt = self.embedding_dropout(self.embedder(tgt))

		# 2. att inputs: keys n values
		att_keys = acous_outputs
		att_vals = acous_outputs

		# generate acous mask: True for trailing 0's
		if type(acous_lens) != type(None):
			# reduce by 8
			lens = torch.cat([elem + 8 - elem % 8 for elem in acous_lens]) / 8
			max_acous_len = acous_outputs.size(1)
			# mask=True over trailing 0s
			mask = torch.arange(max_acous_len).to(device=device).expand(
				batch_size, max_acous_len) >= lens.unsqueeze(1).to(device=device)
		else:
			mask = None

		# 3. init hidden states
		dec_hidden = None

		# 4. run dec + att + shared + output
		"""
			teacher_forcing_ratio = 1.0 -> always teacher forcing
			E.g.:
				acous 	        = [acous_len/8]
				tgt_chunk in    = w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
				predicted       = w1 w2 w3 </s> <pad> <pad> <pad>   [max_seq_len]
		"""

		# LAS under teacher forcing
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

		# beam search decoding
		if not is_training and beam_width > 1:
			decoder_outputs, decoder_hidden, metadata = \
				self.beam_search_decoding(att_keys, att_vals, dec_hidden, mask,
										beam_width=beam_width)
			return decoder_outputs, decoder_hidden, metadata

		# no beam search decoding
		tgt_chunk = self.embedder(torch.Tensor([BOS])
			.repeat(batch_size,1).type(torch.LongTensor).to(device=device)) # BOS
		cell_value = torch.FloatTensor([0])\
			.repeat(batch_size, 1, self.hidden_size_shared).to(device=device)
		prev_c = torch.FloatTensor([0]).repeat(batch_size, 1, max_seq_len).to(device=device)
		attn_outputs = []
		for idx in range(max_seq_len):
			predicted_logsoftmax, dec_hidden, step_attn, c_out, cell_value, attn_output = \
				self.forward_step(self.acous_att, self.acous_ffn, self.acous_out,
								att_keys, att_vals, tgt_chunk, cell_value,
								dec_hidden, mask, prev_c)
			predicted_logsoftmax = predicted_logsoftmax.squeeze(1) # [b, vocab_size]
			step_output = predicted_logsoftmax
			symbols, decoder_outputs, sequence_symbols, lengths = \
				self.decode(idx, step_output, decoder_outputs, sequence_symbols, lengths)
			prev_c = c_out
			if use_teacher_forcing:
				tgt_chunk = emb_tgt[:, idx].unsqueeze(1)
			else:
				tgt_chunk = self.embedder(symbols)
			ret_dict[KEY_ATTN_SCORE].append(step_attn)
			attn_outputs.append(attn_output)

		ret_dict[KEY_SEQUENCE] = sequence_symbols
		ret_dict[KEY_LENGTH] = lengths.tolist()
		ret_dict[KEY_ATTN_OUT] = attn_outputs

		# import pdb; pdb.set_trace()

		return decoder_outputs, dec_hidden, ret_dict


	def decode(self, step, step_output, decoder_outputs, sequence_symbols, lengths):

			"""
				Greedy decoding
				Args:
					step: step idx
					step_output: log predicted_softmax [batch_size, 1, vocab_size_dec]
				Returns:
					symbols: most probable symbol_id [batch_size, 1]
			"""
			decoder_outputs.append(step_output)
			symbols = decoder_outputs[-1].topk(1)[1]
			sequence_symbols.append(symbols)

			eos_batches = torch.max(symbols.data.eq(EOS), symbols.data.eq(PAD))
			# equivalent to logical OR
			# eos_batches = symbols.data.eq(PAD)
			if eos_batches.dim() > 0:
				eos_batches = eos_batches.cpu().view(-1).numpy()
				update_idx = ((lengths > step) & eos_batches) != 0
				lengths[update_idx] = len(sequence_symbols)
			return symbols, decoder_outputs, sequence_symbols, lengths


	def forward_step(self, att_func, ffn_func, out_func,
		att_keys, att_vals, tgt_chunk, prev_cell_value,
		dec_hidden=None, mask_src=None, prev_c=None):

		"""
			manual unrolling

			Args:
				att_keys:   [batch_size, seq_len, acous_hidden_size * 2]
				att_vals:   [batch_size, seq_len, acous_hidden_size * 2]
				tgt_chunk:  tgt word embeddings
							no teacher forcing - [batch_size, 1, embedding_size_dec]
							(becomes 2d when indexed)
				prev_cell_value:
							previous cell value before prediction
							[batch_size, 1, self.state_size]
				dec_hidden:
							initial hidden state for dec layer
				mask_src:
							mask of PAD for src sequences
				prev_c:
							used in hybrid attention mechanism

			Returns:
				predicted_softmax: log probilities [batch_size, vocab_size_dec]
				dec_hidden: a list of hidden states of each dec layer
				attn: attention weights
				cell_value: transformed attention output
							[batch_size, 1, self.hidden_size_shared]
		"""

		# record sizes
		batch_size = tgt_chunk.size(0)
		tgt_chunk_etd = torch.cat([tgt_chunk, prev_cell_value], -1)
		tgt_chunk_etd = tgt_chunk_etd\
			.view(-1, 1, self.embedding_size + self.hidden_size_shared)

		# run dec
		# default dec_hidden: [h_0, c_0];
		# with h_0 [num_layers * num_directions(==1), batch, hidden_size]
		if not self.residual:
			dec_outputs, dec_hidden = self.dec(tgt_chunk, dec_hidden)
			dec_outputs = self.dropout(dec_outputs)
		else:
			# store states layer by layer -
			# num_layers * ([1, batch, hidden_size], [1, batch, hidden_size])
			dec_hidden_lis = []

			# layer0
			dec_func_first = getattr(self.dec, 'l0')
			if type(dec_hidden) == type(None):
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_etd, None)
			else:
				index = torch.tensor([0]).to(device=device) # choose the 0th layer
				dec_hidden_in = tuple(
					[h.index_select(dim=0, index=index) for h in dec_hidden])
				dec_outputs, dec_hidden_out = dec_func_first(tgt_chunk_etd, dec_hidden_in)
			dec_hidden_lis.append(dec_hidden_out)
			# no residual for 0th layer
			dec_outputs = self.dropout(dec_outputs)

			# layer1+
			for i in range(1, self.num_unilstm_dec):
				dec_inputs = dec_outputs
				dec_func = getattr(self.dec, 'l'+str(i))
				if type(dec_hidden) == type(None):
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, None)
				else:
					index = torch.tensor([i]).to(device=device)
					dec_hidden_in = tuple(
						[h.index_select(dim=0, index=index) for h in dec_hidden])
					dec_outputs, dec_hidden_out = dec_func(dec_inputs, dec_hidden_in)
				dec_hidden_lis.append(dec_hidden_out)
				if i < self.num_unilstm_dec - 1:
					dec_outputs = dec_outputs + dec_inputs
				dec_outputs = self.dropout(dec_outputs)

			# convert to tuple
			h_0 = torch.cat([h[0] for h in dec_hidden_lis], 0)
			c_0 = torch.cat([h[1] for h in dec_hidden_lis], 0)
			dec_hidden = tuple([h_0, c_0])

		# run att
		att_func.set_mask(mask_src)
		att_outputs, attn, c_out = att_func(dec_outputs, att_keys, att_vals, prev_c=prev_c)
		att_outputs = self.dropout(att_outputs)

		# run ff + softmax
		ff_inputs = torch.cat((att_outputs, dec_outputs), dim=-1)
		ff_inputs_size = self.acous_hidden_size * 2 + self.hidden_size_dec
		cell_value = ffn_func(ff_inputs.view(-1, 1, ff_inputs_size))
		outputs = out_func(cell_value.contiguous().view(-1, self.hidden_size_shared))
		predicted_logsoftmax = F.log_softmax(outputs, dim=1).view(batch_size, 1, -1)

		return predicted_logsoftmax, dec_hidden, attn, c_out, cell_value, att_outputs


	def beam_search_decoding(self,
		att_keys, att_vals, dec_hidden=None, mask_src=None, prev_c=None, beam_width=10):

		"""
			beam search decoding - only used for evaluation
			Modified from -
			https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py

			Shortcuts:
				beam_width: k
				batch_size: b
				vocab_size: v
				max_seq_len: l

			Args:
				att_keys:   [b x l x acous_hidden_size * 2 + optional key size (key_size)]
				att_vals:   [b x l x acous_hidden_size * 2 (val_size)]
				dec_hidden:
							initial hidden state for dec layer [b x h_dec]
				mask_src:
							mask of PAD for src sequences
				beam_width: beam width kept during searching

			Returns:
				decoder_outputs: output probabilities [(batch, 1, vocab_size)] * T
				decoder_hidden (num_layers * num_direc tions, batch, hidden_size):
					tensor containing the last hidden state of the decoder.
				ret_dict: dictionary containing additional information as follows
				{
					*length*: list of integers representing lengths of output sequences,
					*topk_length*:
						list of integers representing lengths of beam search sequences,
					*sequence*: list of sequences,
						where each sequence is a list of predicted token IDs,
					*topk_sequence*: list of beam search sequences,
						each beam is a list of token IDs,
					*outputs*: [(batch, k, vocab_size)] * sequence_length:
						A list of the output probabilities (p_n)
				}.
		"""

		# define var
		batch_size = att_keys.size(0)
		self.pos_index = Variable(torch.LongTensor(range(batch_size)) * beam_width)\
			.view(-1, 1).to(device=device)

		# initialize the input vector; att_c_value
		input_var = Variable(torch.transpose(
			torch.LongTensor([[BOS] * batch_size * beam_width]), 0, 1)).to(device=device)
		input_var_emb = self.embedder_dec(input_var).to(device=device)
		prev_c = torch.FloatTensor([0])\
			.repeat(batch_size, 1, self.max_seq_len).to(device=device)
		cell_value = torch.FloatTensor([0])\
			.repeat(batch_size, 1, self.hidden_size_shared).to(device=device)

		# inflate attention keys and values (derived from encoder outputs)
		# correct ordering
		inflated_att_keys = att_keys.repeat_interleave(beam_width, dim=0)
		inflated_att_vals = att_vals.repeat_interleave(beam_width, dim=0)
		inflated_mask_src = mask_src.repeat_interleave(beam_width, dim=0)
		inflated_prev_c = prev_c.repeat_interleave(beam_width, dim=0)
		inflated_cell_value = cell_value.repeat_interleave(beam_width, dim=0)

		# inflate hidden states and others
		# note that inflat_hidden_state might be faulty - currently using None so it's fine
		dec_hidden = inflat_hidden_state(dec_hidden, beam_width)

		# Initialize the scores; for the first step,
		# ignore the inflated copies to avoid duplicate entries in the top k
		sequence_scores = torch.Tensor(batch_size * beam_width, 1).to(device=device)
		sequence_scores.fill_(-float('Inf'))
		sequence_scores.index_fill_(0,
			torch.LongTensor([i * beam_width for i in range(0, batch_size)])\
			.to(device=device), 0.0)
		sequence_scores = Variable(sequence_scores)

		# Store decisions for backtracking
		stored_outputs = list()         # raw softmax scores [bk x v] * T
		stored_scores = list()          # topk scores [bk] * T
		stored_predecessors = list()    # preceding beam idx (from 0-bk) [bk] * T
		stored_emitted_symbols = list() # word ids [bk] * T
		stored_hidden = list()          #

		for _ in range(self.max_seq_len):

			predicted_softmax, dec_hidden, step_attn, inflated_c_out, inflated_cell_value, *_ = \
				self.forward_step(self.acous_att, self.acous_ffn, self.acous_out,
					inflated_att_keys, inflated_att_vals, input_var_emb, inflated_cell_value,
					dec_hidden, inflated_mask_src, inflated_prev_c)
			inflated_prev_c = inflated_c_out

			# retain output probs
			stored_outputs.append(predicted_softmax) # [bk x v]

			# To get the full sequence scores for the new candidates,
			# add the local scores for t_i to the predecessor scores for t_(i-1)
			sequence_scores = _inflate(sequence_scores, self.vocab_size, 1)
			sequence_scores += predicted_softmax.squeeze(1) # [bk x v]

			# [b x kv] -> [b x k]
			scores, candidates = sequence_scores.view(batch_size, -1)\
				.topk(beam_width, dim=1)

			# Reshape input = (bk, 1) and sequence_scores = (bk, 1)
			input_var = (candidates % self.vocab_size).view(batch_size * beam_width, 1)\
				.to(device=device)
			input_var_emb = self.embedder_dec(input_var)
			sequence_scores = scores.view(batch_size * beam_width, 1) #[bk x 1]

			# Update fields for next timestep
			predecessors = (candidates / self.vocab_size + self.pos_index\
				.expand_as(candidates)).view(batch_size * beam_width, 1)

			# dec_hidden: [h_0, c_0];
			# with h_0 [num_layers * num_directions, batch, hidden_size]
			if isinstance(dec_hidden, tuple):
				dec_hidden = tuple(
					[h.index_select(1, predecessors.squeeze()) for h in dec_hidden])
			else:
				dec_hidden = dec_hidden.index_select(1, predecessors.squeeze())

			# Update sequence scores and erase scores
			# for end-of-sentence symbol so that they aren't expanded
			stored_scores.append(sequence_scores.clone())

			# Cache results for backtracking
			stored_predecessors.append(predecessors)
			stored_emitted_symbols.append(input_var)
			stored_hidden.append(dec_hidden)

		# Do backtracking to return the optimal values
		output, h_t, h_n, s, l, p = self._backtrack(stored_outputs, stored_hidden,
			stored_predecessors, stored_emitted_symbols,
			stored_scores, batch_size, self.hidden_size_dec, beam_width)

		# Build return objects
		decoder_outputs = [step[:, 0, :].squeeze(1) for step in output]
		if isinstance(h_n, tuple):
			decoder_hidden = tuple([h[:, :, 0, :] for h in h_n])
		else:
			decoder_hidden = h_n[:, :, 0, :]
		metadata = {}
		metadata['output'] = output
		metadata['h_t'] = h_t
		metadata['score'] = s
		metadata['topk_length'] = l
		metadata['topk_sequence'] = p # [b x k x 1] * T
		metadata['length'] = [seq_len[0] for seq_len in l]
		metadata['sequence'] = [seq[:, 0] for seq in p]

		return decoder_outputs, decoder_hidden, metadata


	def _backtrack(self,
		nw_output, nw_hidden, predecessors, symbols,
		scores, b, hidden_size, beam_width):

		"""
			Backtracks over batch to generate optimal k-sequences.
			https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/TopKDecoder.py

			Args:
				nw_output [(batch*k, vocab_size)] * sequence_length:
					A Tensor of outputs from network
				nw_hidden [(num_layers, batch*k, hidden_size)] * sequence_length:
					A Tensor of hidden states from network
				predecessors [(batch*k)] * sequence_length: A Tensor of predecessors
				symbols [(batch*k)] * sequence_length: A Tensor of predicted tokens
				scores [(batch*k)] * sequence_length:
					A Tensor containing sequence scores for every token
					t = [0, ... , seq_len - 1]
				b: Size of the batch
				hidden_size: Size of the hidden state

			Returns:
				output [(batch, k, vocab_size)] * sequence_length:
					A list of the output probabilities (p_n)
				from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
				h_t [(batch, k, hidden_size)] * sequence_length:
					A list containing the output features (h_n)
					from the last layer of the RNN, for every n = [0, ... , seq_len - 1]
				h_n(batch, k, hidden_size):
					A Tensor containing the last hidden state for all top-k sequences.
				score [batch, k]:
					A list containing the final scores for all top-k sequences
				length [batch, k]:
					A list specifying the length of each sequence in the top-k candidates
				p (batch, k, sequence_len):
					A Tensor containing predicted sequence [b x k x 1] * T
		"""

		# initialize return variables given different types
		output = list()
		h_t = list()
		p = list()

		# Placeholder for last hidden state of top-k sequences.
		# If a (top-k) sequence ends early in decoding, `h_n` contains
		# its hidden state when it sees EOS.  Otherwise, `h_n` contains
		# the last hidden state of decoding.
		lstm = isinstance(nw_hidden[0], tuple)
		if lstm:
			state_size = nw_hidden[0][0].size()
			h_n = tuple([torch.zeros(state_size).to(device=device),
				torch.zeros(state_size).to(device=device)])
		else:
			h_n = torch.zeros(nw_hidden[0].size()).to(device=device)

		# Placeholder for lengths of top-k sequences
		# Similar to `h_n`
		l = [[self.max_seq_len] * beam_width for _ in range(b)]

		# the last step output of the beams are not sorted
		# thus they are sorted here
		sorted_score, sorted_idx = scores[-1].view(b, beam_width).topk(beam_width)
		sorted_score = sorted_score.to(device=device)
		sorted_idx = sorted_idx.to(device=device)

		# initialize the sequence scores with the sorted last step beam scores
		s = sorted_score.clone().to(device=device)

		batch_eos_found = [0] * b   # the number of EOS found
									# in the backward loop below for each batch

		t = self.max_seq_len - 1
		# initialize the back pointer with the sorted order of the last step beams.
		# add self.pos_index for indexing variable with b*k as the first dimension.
		t_predecessors = (sorted_idx + self.pos_index.expand_as(sorted_idx))\
			.view(b * beam_width).to(device=device)

		while t >= 0:
			# Re-order the variables with the back pointer
			current_output = nw_output[t].index_select(0, t_predecessors)
			if lstm:
				current_hidden = tuple(
					[h.index_select(1, t_predecessors) for h in nw_hidden[t]])
			else:
				current_hidden = nw_hidden[t].index_select(1, t_predecessors)
			current_symbol = symbols[t].index_select(0, t_predecessors)

			# Re-order the back pointer of the previous step with the back pointer of
			# the current step
			t_predecessors = predecessors[t]\
				.index_select(0, t_predecessors).squeeze().to(device=device)

			"""
				This tricky block handles dropped sequences that see EOS earlier.
				The basic idea is summarized below:

				  Terms:
					  Ended sequences = sequences that see EOS early and dropped
					  Survived sequences = sequences in the last step of the beams

					  Although the ended sequences are dropped during decoding,
				  their generated symbols and complete backtracking information are still
				  in the backtracking variables.
				  For each batch, everytime we see an EOS in the backtracking process,
					  1. If there is survived sequences in the return variables, replace
					  the one with the lowest survived sequence score with the new ended
					  sequences
					  2. Otherwise, replace the ended sequence with the lowest sequence
					  score with the new ended sequence
			"""

			eos_indices = symbols[t].data.squeeze(1).eq(EOS).nonzero().to(device=device)
			if eos_indices.dim() > 0:
				for i in range(eos_indices.size(0)-1, -1, -1):
					# Indices of the EOS symbol for both variables
					# with b*k as the first dimension, and b, k for
					# the first two dimensions
					idx = eos_indices[i]
					b_idx = int(idx[0] / beam_width)
					# The indices of the replacing position
					# according to the replacement strategy noted above
					res_k_idx = beam_width - (batch_eos_found[b_idx] % beam_width) - 1
					batch_eos_found[b_idx] += 1
					res_idx = b_idx * beam_width + res_k_idx

					# Replace the old information in return variables
					# with the new ended sequence information
					t_predecessors[res_idx] = predecessors[t][idx[0]].to(device=device)
					current_output[res_idx, :] = nw_output[t][idx[0], :].to(device=device)
					if lstm:
						current_hidden[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :]\
							.to(device=device)
						current_hidden[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :]\
							.to(device=device)
						h_n[0][:, res_idx, :] = nw_hidden[t][0][:, idx[0], :].data\
							.to(device=device)
						h_n[1][:, res_idx, :] = nw_hidden[t][1][:, idx[0], :].data\
							.to(device=device)
					else:
						current_hidden[:, res_idx, :] = nw_hidden[t][:, idx[0], :]\
							.to(device=device)
						h_n[:, res_idx, :] = nw_hidden[t][:, idx[0], :].data\
							.to(device=device)
					current_symbol[res_idx, :] = symbols[t][idx[0]].to(device=device)
					s[b_idx, res_k_idx] = scores[t][idx[0]].data[0].to(device=device)
					l[b_idx][res_k_idx] = t + 1

			# record the back tracked results
			output.append(current_output)
			h_t.append(current_hidden)
			p.append(current_symbol)

			t -= 1

		# Sort and re-order again as the added ended sequences may change
		# the order (very unlikely)
		s, re_sorted_idx = s.topk(beam_width)
		for b_idx in range(b):
			l[b_idx] = [l[b_idx][k_idx.item()] for k_idx in re_sorted_idx[b_idx,:]]

		re_sorted_idx = (re_sorted_idx + self.pos_index.expand_as(re_sorted_idx))\
			.view(b * beam_width).to(device=device)

		# Reverse the sequences and re-order at the same time
		# It is reversed because the backtracking happens in reverse time order
		output = [step.index_select(0, re_sorted_idx)\
			.view(b, beam_width, -1) for step in reversed(output)]
		p = [step.index_select(0, re_sorted_idx)\
			.view(b, beam_width, -1) for step in reversed(p)]
		if lstm:
			h_t = [tuple([h.index_select(1, re_sorted_idx.to(device=device))\
				.view(-1, b, beam_width, hidden_size) for h in step]) for step in reversed(h_t)]
			h_n = tuple([h.index_select(1, re_sorted_idx.data.to(device=device))\
				.view(-1, b, beam_width, hidden_size) for h in h_n])
		else:
			h_t = [step.index_select(1, re_sorted_idx.to(device=device))\
				.view(-1, b, beam_width, hidden_size) for step in reversed(h_t)]
			h_n = h_n.index_select(1, re_sorted_idx.data.to(device=device))\
				.view(-1, b, beam_width, hidden_size)
		s = s.data

		return output, h_t, h_n, s, l, p


def get_base_hidden(hidden):

	""" strip the nested tuple, get the last hidden state """

	tuple_dim = []
	while isinstance(hidden, tuple):
		tuple_dim.append(len(hidden))
		hidden = hidden[-1]
	return hidden, tuple_dim


def _inflate(tensor, times, dim):

	"""
		Given a tensor, 'inflates' it along the given dimension
		by replicating each slice specified number of times (in-place)
		Args:
			tensor: A :class:`Tensor` to inflate
			times: number of repetitions
			dim: axis for inflation (default=0)
		Returns:
			A :class:`Tensor`
		Examples::
			>> a = torch.LongTensor([[1, 2], [3, 4]])
			>> a
			1   2
			3   4
			[torch.LongTensor of size 2x2]
			>> b = ._inflate(a, 2, dim=1)
			>> b
			1   2   1   2
			3   4   3   4
			[torch.LongTensor of size 2x4]
			>> c = _inflate(a, 2, dim=0)
			>> c
			1   2
			3   4
			1   2
			3   4
			[torch.LongTensor of size 4x2]
	"""

	repeat_dims = [1] * tensor.dim()
	repeat_dims[dim] = times
	return tensor.repeat(*repeat_dims)


def inflat_hidden_state(hidden_state, k):

	if hidden_state is None:
		hidden = None
	else:
		if isinstance(hidden_state, tuple):
			hidden = tuple([_inflate(h, k, 1) for h in hidden_state])
		else:
			hidden = _inflate(hidden_state, k, 1)
	return hidden
