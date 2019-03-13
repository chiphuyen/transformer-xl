80712 lm-data/test.txt
40256704 lm-data/train.txt
200000 lm-data/train.vocab
80844 lm-data/valid.txt

Character per word: average = 4.337079941775207.
Word count per line: average = 19.874401545242176, median = 16.0, max = 600, min = 1, stddev = 16.018985036617668
Vocab size is: 200000. 1871827 unk tokens out of 800069004: 0.002339581949358958


medium: 6582.35s / 4000 batches
-> 295740.753989s = 82h per epoch

small: 3972.96s / 4000 batches
-> 50h per epoch


cutoffs = [20000, 40000, 140000]

LM-TFM-libri/20190306-105034
1 epoch - valid ppl 78.5

LM-TFM-libri/20190307-135407
6 epoch - valid ppl 65.152

| Eval  28 at step   112000 | time: 2539.84s | valid loss  4.18 | valid ppl    65.152


LM-TFM-libri/20190308-114428
n_all_param : 52,861,525
n_nonemb_param : 37785,600


Best eval ppl
| Eval 116 at step   464000 | time: 3124.36s | valid loss  4.08 | valid ppl    59.407 (epoch 25)


Test ppl
| End of training | test loss  4.08 | test ppl    59.278


WSJ
Vocab size is: 162318. 0 unk tokens out of 37,242,497: 0.0

Threshold = 3
16179 wsj/test.txt
  1592831 wsj/train.txt
    85547 wsj/train.vocab
    16526 wsj/valid.txt
Vocab size is: 85547. 96284 unk tokens out of 36493781: 0.0026383673426439427

Character per word: average = 5.026113501465812.
Word count per line: average = 22.910902619197607, median = 22.0, max = 223, min = 1, stddev = 11.227452595303074.

Base:
params = 40133211
non emb params = 32823552

Folder:
LM-TFM-wsj/20190312-154911
best checkpoint is saved in model.pt and optimizer.pt

There are more checkpoints correspoding to different eval ppl, with each model.pt and optimizer.pt stored in a folder named by ppl.

- n_all_param : 40133211
- n_nonemb_param : 32823552

Eval:
Eval  20 at step    20000 | time: 522.29s | valid loss  3.83 | valid ppl    46.137 (epoch 9)

Test:

| test loss  3.81 | test ppl    45.295 

==================================================================================================================================
To ignore:
900k vocab
971915
924 unk tokens out of 1609255 - 0.000574178734880426
860 unk tokens out of 1610470 - 0.0005340056008494414

Vocab size is: 419430. 677693 unk tokens out of 800069004: 0.000847043188289794
1774 unk tokens out of 1609255 - 0.0011023734585258395
1757 unk tokens out of 1610470 - 0.0010909858612703125
677693 unk tokens out of 800069004 - 0.000847043188289794

Good numbers:

--batch_size 1536 \
--multi_gpu \
--gpu0_bsz 1024 \