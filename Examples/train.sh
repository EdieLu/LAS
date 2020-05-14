#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
echo $CUDA_VISIBLE_DEVICES

# python 3.6 
# pytorch 1.1
source activate py13-cuda9

# ------------------------ hyper param --------------------------
max_seq_len=90 # 400 for char | 90 for  word | 110 for bpe
batch_size=256
minibatch_partition=40

# print_every=1
# checkpoint_every=2
print_every=150
checkpoint_every=900
num_epochs=100
learning_rate=0.001

random_seed=25
eval_with_mask=True
savedir=acous-las-models-v3/debug/
loaddir=None
# loaddir=acous-las-models-v3/las-word-v002-cont/checkpoints/2020_04_17_01_56_38/ 

# ------------------------ file dir --------------------------
# use_type=char # char | word | bpe
# train_path_src=lib-bpe/swbd-asr/train/text.proc
# dev_path_src=lib-bpe/swbd-asr/valid/text.proc
# path_vocab=lib-bpe/vocab/char.en
# load_embedding=None
# embedding_size=50

# use_type=bpe # bpe
# train_path_src=lib-bpe/swbd-asr/train/text.proc.bpe
# dev_path_src=lib-bpe/swbd-asr/valid/text.proc.bpe
# path_vocab=lib-bpe/vocab/bpe_en_25000+pad.txt
# load_embedding=None
# embedding_size=200

# use_type=word # word
# train_path_src=lib-bpe/swbd-asr/train/text.proc
# dev_path_src=lib-bpe/swbd-asr/valid/text.proc
# path_vocab=lib/vocab/swbd-min1.en
# load_embedding=lib/embeddings/glove.6B.200d.txt
# embedding_size=200

use_type=word # word-v2
train_path_src=lib-bpe/swbd-asr/train/text.proc.split
dev_path_src=lib-bpe/swbd-asr/valid/text.proc.split
path_vocab=lib-bpe/vocab/swbdasr-min1.en
load_embedding=lib/embeddings/glove.6B.200d.txt
embedding_size=200

train_acous_path=lib-bpe/swbd-asr/train/flis
dev_acous_path=lib-bpe/swbd-asr/valid/flis

# ------------------------ model config --------------------------
batch_norm=False
acous_norm=True
spec_aug=True
enc_mode=pyramid # cnn

acous_hidden_size=256
# enc = 4 layer blstm
acous_att_mode=bilinear
hidden_size_dec=200
hidden_size_shared=200
num_unilstm_dec=4


export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/acous-las-v3/train.py \
	--train_path_src $train_path_src \
	--train_acous_path $train_acous_path \
	--dev_path_src $dev_path_src \
	--dev_acous_path $dev_acous_path \
	--use_type $use_type \
	--path_vocab_src $path_vocab \
	--load_embedding $load_embedding \
	--save $savedir \
	--load $loaddir \
	--random_seed $random_seed \
	--embedding_size $embedding_size \
	--acous_hidden_size $acous_hidden_size \
	--acous_att_mode $acous_att_mode \
	--hidden_size_dec $hidden_size_dec \
	--hidden_size_shared $hidden_size_shared \
	--num_unilstm_dec $num_unilstm_dec \
	--residual True \
	--max_seq_len $max_seq_len \
	--batch_size $batch_size \
	--batch_first True \
	--eval_with_mask True \
	--scheduled_sampling False \
	--teacher_forcing_ratio 1.0 \
	--dropout 0.2 \
	--embedding_dropout 0.0 \
	--num_epochs $num_epochs \
	--use_gpu True \
	--max_grad_norm 1.0 \
	--learning_rate $learning_rate \
	--spec_aug $spec_aug \
	--acous_norm $acous_norm \
	--batch_norm $batch_norm \
	--enc_mode $enc_mode \
	--checkpoint_every $checkpoint_every \
	--print_every $print_every \
	--minibatch_partition $minibatch_partition \

	# bahdanau / hybrid	
