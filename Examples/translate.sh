#!/bin/bash
#$ -S /bin/bash

echo $HOSTNAME
unset LD_PRELOAD # when run on the stack it has /usr/local/grid/agile/admin/cudadevice.so which will give core dumped
export PATH=/home/mifs/ytl28/anaconda3/bin/:$PATH

# export CUDA_VISIBLE_DEVICES=$X_SGE_CUDA_DEVICE
export CUDA_VISIBLE_DEVICES=0
echo $CUDA_VISIBLE_DEVICES 

# python 3.6
# pytorch 1.1
source activate py13-cuda9

# ===========================================
# fname=test_swbddev
# ftst=lib-bpe/swbd-asr/valid/text.proc
# acous_path=lib-bpe/swbd-asr/valid/flis
# seqlen=400

# fname=test_swbd
# ftst=lib-bpe/swbd-asr/test/text.proc
# acous_path=lib-bpe/swbd-asr/test/flis
# seqlen=400

# fname=test_eval2000_ch
# ftst=lib-bpe/swbd-asr/eval2000-ch/text
# acous_path=lib-bpe/swbd-asr/eval2000-ch/flis
# # seqlen=270 # char
# seqlen=60 # word

# fname=test_eval2000_sw
# ftst=lib-bpe/swbd-asr/eval2000-sw/text
# acous_path=lib-bpe/swbd-asr/eval2000-sw/flis
# # seqlen=350 # char
# seqlen=80 # word

# fname=test_eval3_tf
# ftst=lib/add-acoustics/eval3/test_srctgt/src.txt
# acous_path=lib/add-acoustics/eval3/test_srctgt/feat/flis
# # seqlen=730 #char
# seqlen=145 #word

fname=test_swbd_tf
ftst=lib/add-acoustics/swbd/align/new_test_srctgt/asr/src.txt
acous_path=lib/add-acoustics/swbd/align/new_test_srctgt/asr/feat/flis
# seqlen=360 #char
seqlen=80 #word

# ----- models ------
path_vocab=lib/vocab/swbd-min1.en
use_type=word
acous_norm=True
model=acous-las-models/las-word/
ckpt=2020_04_14_03_40_19

# path_vocab=lib-bpe/vocab/swbdasr-min1.en [not active]
# use_type=word
# acous_norm=True
# model=acous-las-models-v3/las-word-v002-cont/
# ckpt=2020_04_18_11_02_33

# path_vocab=lib-bpe/vocab/bpe_en_25000+pad.txt
# use_type=bpe
# acous_norm=True
# model=acous-las-models/las-bpe/
# ckpt=2020_04_15_19_19_26

# path_vocab=lib-bpe/vocab/char.en
# use_type=char
# acous_norm=False
# model=acous-las-models/las-char/
# ckpt=2020_04_11_17_51_40

batch_size=50
# mode=2
# use_gpu=True
mode=6
use_gpu=False

export PYTHONBIN=/home/mifs/ytl28/anaconda3/envs/py13-cuda9/bin/python3
$PYTHONBIN /home/alta/BLTSpeaking/exp-ytl28/local-ytl/acous-las-v3/translate.py \
    --test_path_src $ftst \
    --test_acous_path $acous_path \
    --path_vocab_src $path_vocab \
    --use_type $use_type \
    --acous_norm $acous_norm \
    --load $model/checkpoints/$ckpt \
    --test_path_out $model/$fname/$ckpt/ \
    --max_seq_len $seqlen \
    --batch_size $batch_size \
    --use_gpu $use_gpu \
    --beam_width 1 \
    --eval_mode $mode 


