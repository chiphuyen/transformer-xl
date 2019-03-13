#!/bin/bash

if [[ $1 == 'train' ]]; then
    echo 'Run training...'
    python train.py \
        --cuda \
        --data /home/chipn/data/wsj-85k/ \
        --dataset wsj \
        --adaptive \
        --n_layer 8 \
        --d_model 448 \
        --div_val 4 \
        --n_head 8 \
        --d_head 36 \
        --d_inner 1024 \
        --dropout 0.0 \
        --dropatt 0.0 \
        --optim adam \
        --warmup_step 0 \
        --max_step 200000 \
        --lr 0.00025 \
        --tgt_len 32 \
        --mem_len 32 \
        --eval_tgt_len 32 \
        --eval-interval 1000 \
        --batch_size 512 \
        --multi_gpu \
        --gpu0_bsz 512 \
        ${@:2}
elif [[ $1 == 'eval' ]]; then
    echo 'Run evaluation...'
    python eval.py \
        --cuda \
        --data /home/chipn/data/wsj-85k/ \
        --work_dir LM-TFM-libri/20190307-135407 \
        --dataset libri \
        --batch_size 1 \
        --tgt_len 32 \
        --mem_len 128 \
        --split test \
        --same_length \
        ${@:2}
elif [[ $1 == 'score' ]]; then
    echo 'Score...'
    python score.py \
        --cuda \
        --data /home/chipn/data/jasper/ \
        --vocab_file /home/chipn/data/librispeech/lm-data-unk/1b_word_vocab.txt \
        --work_dir LM-TFM-libri/20190307-135407 \
        --tgt_len 32 \
        --mem_len 128 \
        --same_length \
        ${@:2}
else
    echo 'unknown argment 1'
fi
