# coding: utf-8
import argparse
import time
import math
import os, sys

import torch

# from data_utils import get_lm_corpus
from mem_transformer import MemTransformerLM
from data_utils import LMOrderedIterator
from utils.exp_utils import get_logger
from utils.vocabulary import Vocab

parser = argparse.ArgumentParser(description='PyTorch Transformer Language Model')
parser.add_argument('--data', type=str, default='../data/wikitext-103',
                    help='location of the data corpus')
parser.add_argument('--vocab_file', type=str, default='',
                    help='location to the vocabulary file')
parser.add_argument('--tgt_len', type=int, default=5,
                    help='number of tokens to predict')
parser.add_argument('--ext_len', type=int, default=0,
                    help='length of the extended context')
parser.add_argument('--mem_len', type=int, default=0,
                    help='length of the retained previous heads')
parser.add_argument('--clamp_len', type=int, default=-1,
                    help='max positional embedding index')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--work_dir', type=str, required=True,
                    help='path to the work_dir')
parser.add_argument('--no_log', action='store_true',
                    help='do not log the eval result')
parser.add_argument('--same_length', action='store_true',
                    help='set same length attention with masking')
args = parser.parse_args()
assert args.ext_len >= 0, 'extended context length must be non-negative'

device = torch.device("cuda" if args.cuda else "cpu")

# Get logger
logging = get_logger(os.path.join(args.work_dir, 'log.txt'),
                     log_=not args.no_log)

# Load the best saved model.
with open(os.path.join(args.work_dir, 'model.pt'), 'rb') as f:
    model = torch.load(f)
model.backward_compatible()
model = model.to(device)

logging('Scoring with tgt_len {} ext_len {} mem_len {} clamp_len {}'.format(
       args.tgt_len, args.ext_len, args.mem_len, args.clamp_len))

model.reset_length(args.tgt_len, args.ext_len, args.mem_len)
if args.clamp_len > 0:
    model.clamp_len = args.clamp_len
if args.same_length:
    model.same_length = True



# Load dataset
# strings = ["a barrel's the jolliest bed going on the tramp i mean", "a bit late to secure accommodations isn't it"]
strings = ["eat don't the the the don't flower the", "i hate school", "i love school", "i love my mom", "i love my dad", "she's an engineer", "he's an engineer", "she's a nurse", "he's a nurse", "she's a manager", "he's a manager"]
vocab = Vocab(vocab_file=args.vocab_file)
vocab.build_vocab()
sents = vocab.encode_sents([['<S>'] + string.strip().lower().split() + ['<S>'] for string in strings])
device = torch.device('cuda' if args.cuda else 'cpu')

###############################################################################
# Scoring code
###############################################################################

def score(sents, device):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    log_losses = []
    start_time = time.time()
    with torch.no_grad():
        for sent in sents:
            sent = sent[:, None].to(device)
            mems = tuple()
            ret = model(sent[:-1], sent[1:], *mems)
            loss, mems = ret[0], ret[1:]
            """
            loss is a tensor of negative log softmax at each predicted token.
            the sum of negative log softmax at all steps will favor long sentences
            it's better to use mean
            """
            log_losses.append(loss.mean().item())
            logging('Took {:.2f}s'.format(time.time() - start_time))
            start_time = time.time()
    return log_losses

# TODO get score_iter from txt
log_losses = score(sents, device)

for i in range(len(strings)):
    print(strings[i])
    print(log_losses[i])
# loss is the negative log softmax


# def format_log(loss, split):
#     if args.dataset in ['enwik8', 'text8']:
#         log_str = '| {0} loss {1:5.2f} | {0} bpc {2:9.5f} '.format(
#             split, loss, loss / math.log(2))
#     else:
#         log_str = '| {0} loss {1:5.2f} | {0} ppl {2:9.3f} '.format(
#             split, loss, math.exp(loss))
#     return log_str

# log_str = ''
# if valid_loss is not None:
#     log_str += format_log(valid_loss, 'valid')
# if test_loss is not None:
#     log_str += format_log(test_loss, 'test')

# logging('=' * 100)
# logging(log_str)
# logging('=' * 100)
