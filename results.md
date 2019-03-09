80712 lm-data/test.txt
40256704 lm-data/train.txt
200000 lm-data/train.vocab
80844 lm-data/valid.txt


Vocab size is: 200000. 1871827 unk tokens out of 800069004: 0.002339581949358958
bz = 224
=> 179717.428571 batches per epoch

117200 batches?

medium: 6582.35s / 4000 batches
-> 295740.753989s = 82h per epoch

small: 3972.96s / 4000 batches
-> 50h per epoch


cutoffs = [20000, 40000, 140000]
1 epoch - valid ppl 78.5
LM-TFM-libri/20190306-105034

6 epoch - valid ppl 65.152
LM-TFM-libri/20190307-135407

| Eval  28 at step   112000 | time: 2539.84s | valid loss  4.18 | valid ppl    65.152

cutoffs = [16000, 32000, 140000]

tensor([   6, 1646,  536])


900k vocab
971915
924 unk tokens out of 1609255 - 0.000574178734880426
860 unk tokens out of 1610470 - 0.0005340056008494414

Vocab size is: 419430. 677693 unk tokens out of 800069004: 0.000847043188289794
1774 unk tokens out of 1609255 - 0.0011023734585258395
1757 unk tokens out of 1610470 - 0.0010909858612703125
677693 unk tokens out of 800069004 - 0.000847043188289794

