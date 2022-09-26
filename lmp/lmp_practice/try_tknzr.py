
#%%
# import tokenizer module
import lmp.tknzr
import lmp.util.tknzr
import lmp.dset
chartknzr = lmp.tknzr.CharTknzr()
wstknzr = lmp.tknzr.WsTknzr()
text = 'Mary has a little lamb.'
print(f'Char Tknzr Result: {chartknzr.tknz(txt = text)}')
print(f'Whitespace Tknzr Result: {wstknzr.tknz(txt = text)}')

EXPNAME = 'ch_poem_tknzr'
# train a tokenizer (first run the following script)
"""
script version:
python3 -m lmp.script.train_tknzr whitespace --exp_name ch_poem_tknzr --dset_name chinese-poem --max_vocab 2000


usage: python -m lmp.script.train_tknzr character [-h] [--dset_name {chinese-poem,demo,WNLI,wiki-text-2}]
                                                   [--exp_name EXP_NAME] [--seed SEED] [--ver VER]
                                                   [--max_vocab MAX_VOCAB] [--min_count MIN_COUNT] [--is_uncased]


"""
# Then check under language-model-playground/exp the trained tokenizer file


# tokenize text
"""
script version:
python -m lmp.script.tknz_txt --exp_name my_tknzr_exp --txt "Hello World"

"""
text = '世界燦爛盛大 歡迎回家！'
ch_poem_tknzr = lmp.util.tknzr.load(exp_name=EXPNAME)
toktext = ch_poem_tknzr.tknz(txt = text)
print(toktext)
print('detokenized:', ch_poem_tknzr.dtknz(toktext))
# =============== special tokens ======================
# special tokens are treated as 1 unit no matter what

from lmp.vars import SP_TKS
print(f'special tokens: {SP_TKS}')
text2 = '<bos>'+ text + '<pad>'
print(ch_poem_tknzr.tknz(txt = text2))

# ===================text normalization ===================
# initialize an uncased tokenizer
uncased_tknzr = lmp.tknzr.CharTknzr(is_uncased = True)
text = 'ABcDEf'
toktext = uncased_tknzr.tknz(txt = text)
print(uncased_tknzr.dtknz(toktext)) # abcdef

# ================== build vocab ===================
Dset = lmp.dset.DemoDset(ver = 'train')
print(Dset[0])
uncased_tknzr.build_vocab(batch_txt = Dset)
print(uncased_tknzr.tk2id)

# %%
dataset =  lmp.dset.WikiText2Dset()
bpetknzr = lmp.tknzr.BPETknzr()
bpetknzr.build_vocab(batch_txt = dataset)
print(bpetknzr.enc('ABcDEf'))

# %%
from lmp.tknzr import BPETknzr
tknzr = BPETknzr()
assert tknzr.tknz('abc def') == ['abc<eow>', 'def<eow>']
assert tknzr.dtknz(['abc<eow>', 'def<eow>']) == 'abc def'

# %%
