#%%
import lmp.tknzr
import lmp.util.tknzr
import lmp.dset


EXPNAME = 'ch_poem_tknzr'
tokenizer = lmp.util.tknzr.load(exp_name=EXPNAME)
# ============== encode =============
print(tokenizer.id2tk.values())
print(len(tokenizer.id2tk)) # max_voab is set to 2000
# ============= pad_to_max ============
batch_text  = [
    '南轅北轍',
    '瘦馬上高山',
    '曲終人散',
]
for text in batch_text:
    print('Original text:',  text)
    tkids = tokenizer.enc(txt = text)
    padded_tkids = tokenizer.pad_to_max(tkids = tkids, max_seq_len = 10)
    print('Encoded text:', padded_tkids)

# %%
# train a byte-pair encoding (a list of subwords)
tknzr = 