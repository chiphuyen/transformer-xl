# Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context

This repository contains the code in both **PyTorch** and **TensorFlow** for our paper
>[Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](http://arxiv.org/abs/1901.02860)

>Zihang Dai\*, Zhilin Yang\*, Yiming Yang, Jaime Carbonell, Quoc V. Le, Ruslan Salakhutdinov (\*: equal contribution)

>Preprint 2018

## TensorFlow

- The source code is in the `tf/` folder, supporting (1) single-node multi-gpu training, and (2) multi-host TPU training.
- Besides the source code, we also provide pretrained "TensorFlow" models with state-of-the-art (SoTA) performances reported in the paper.
- Please refer to `tf/README.md` for details.

## PyTorch

- The source code is in the `pytorch/` folder, supporting single-node multi-gpu training via the module `nn.DataParallel`.
- Please refer to `pytorch/README.md` for details.

## Results

Transformer-XL achieves new state-of-the-art results on multipole language modeling benchmarks. Transformer-XL is also the first to break through the 1.0 barrier on char-level language modeling. Below is a summary.

Method | enwiki8 | text8 | One Billion Word | WT-103 | PTB (w/o finetuning)
-- | -- | -- | -- | -- | -- 
Previous Best | 1.06 | 1.13 | 23.7 | 20.5 | 55.5
Transformer-XL | **0.99** | **1.08** | **21.8** | **18.3** | **54.5**



## Acknowledgement

A large portion of the `getdata.sh` script comes from the [awd-lstm](https://github.com/salesforce/awd-lstm-lm/) repo. Happy Language Modeling :)

CUDA_VISIBLE_DEVICES=1 python main.py --save CH76-EXP3.pt --model=LSTM --emsize=224 --nlayers=4 --lr=20 --nhid=448 --bptt=128 --clip=1.6059587641691437 --dropout=0.15484757015252373 --dropouth=0.0015277383930061283 --dropoute=0.08866500895563889 --dropouti=0.763139048551762 --wdrop=0.02628840032858744

----------------------------------------------------------------------------------------------------
| epoch   9 step    72200 |    984 batches | lr 0.000178 | ms/batch 933.75 | loss  2.99 | ppl    19.898
| epoch   9 step    72400 |   1184 batches | lr 0.000178 | ms/batch 898.73 | loss  3.02 | ppl    20.506
| epoch   9 step    72600 |   1384 batches | lr 0.000177 | ms/batch 898.08 | loss  3.01 | ppl    20.358
| epoch   9 step    72800 |   1584 batches | lr 0.000177 | ms/batch 898.04 | loss  3.00 | ppl    20.165
| epoch   9 step    73000 |   1784 batches | lr 0.000176 | ms/batch 898.50 | loss  2.98 | ppl    19.753
| epoch   9 step    73200 |   1984 batches | lr 0.000176 | ms/batch 898.24 | loss  3.00 | ppl    20.030
| epoch   9 step    73400 |   2184 batches | lr 0.000176 | ms/batch 898.36 | loss  3.00 | ppl    19.986
| epoch   9 step    73600 |   2384 batches | lr 0.000175 | ms/batch 898.55 | loss  3.01 | ppl    20.209
| epoch   9 step    73800 |   2584 batches | lr 0.000175 | ms/batch 898.42 | loss  3.06 | ppl    21.433
| epoch   9 step    74000 |   2784 batches | lr 0.000175 | ms/batch 898.35 | loss  3.03 | ppl    20.604
| epoch   9 step    74200 |   2984 batches | lr 0.000174 | ms/batch 898.61 | loss  3.00 | ppl    20.092
| epoch   9 step    74400 |   3184 batches | lr 0.000174 | ms/batch 898.15 | loss  3.01 | ppl    20.361
| epoch   9 step    74600 |   3384 batches | lr 0.000174 | ms/batch 899.05 | loss  3.01 | ppl    20.260
| epoch   9 step    74800 |   3584 batches | lr 0.000173 | ms/batch 898.61 | loss  3.00 | ppl    20.048
| epoch   9 step    75000 |   3784 batches | lr 0.000173 | ms/batch 898.68 | loss  2.98 | ppl    19.680
| epoch   9 step    75200 |   3984 batches | lr 0.000172 | ms/batch 898.31 | loss  2.94 | ppl    19.000
| epoch   9 step    75400 |   4184 batches | lr 0.000172 | ms/batch 898.82 | loss  2.98 | ppl    19.674
| epoch   9 step    75600 |   4384 batches | lr 0.000172 | ms/batch 898.48 | loss  2.98 | ppl    19.596
| epoch   9 step    75800 |   4584 batches | lr 0.000171 | ms/batch 898.67 | loss  2.97 | ppl    19.517
| epoch   9 step    76000 |   4784 batches | lr 0.000171 | ms/batch 898.32 | loss  2.99 | ppl    19.968
----------------------------------------------------------------------------------------------------
| Eval  19 at step    76000 | time: 3600.91s | valid loss  2.53 | valid ppl    12.561

  - dataset : wt103
    - n_layer : 16
    - n_head : 10
    - d_head : 41
    - d_embed : 410
    - d_model : 410
    - d_inner : 2100
    - dropout : 0.1
    - dropatt : 0.0
    - init : normal
    - emb_init : normal
    - init_range : 0.1
    - emb_init_range : 0.01
    - init_std : 0.02
    - proj_init_std : 0.01
    - optim : adam
    - lr : 0.00025
    - mom : 0.0
    - scheduler : cosine
    - warmup_step : 0
    - decay_rate : 0.5
    - lr_min : 0.0
    - clip : 0.25
    - clip_nonemb : False
    - max_step : 200000
    - batch_size : 60
    - batch_chunk : 1
    - tgt_len : 150
    - eval_tgt_len : 150
    - ext_len : 0
    - mem_len : 150
    - not_tied : False
    - seed : 1111
    - cuda : True
    - adaptive : True
    - div_val : 1
    - pre_lnorm : False
    - varlen : False
    - multi_gpu : True
    - log_interval : 200
    - eval_interval : 4000
    - work_dir : LM-TFM-wt103/20190130-155653
    - restart : False
    - restart_dir : 
    - debug : False
    - same_length : False
    - attn_type : 0
    - clamp_len : -1
    - eta_min : 0.0
    - gpu0_bsz : 4
    - max_eval_steps : -1
    - sample_softmax : -1
    - patience : 0
    - finetune_v2 : False
    - finetune_v3 : False
    - fp16 : False
    - static_loss_scale : 1
    - dynamic_loss_scale : False
    - tied : True
    - n_token : 92654
    - n_all_param : 79149247
    - n_nonemb_param : 41066400
