#%%
import lmp.tknzr
import lmp.util.tknzr
import lmp.dset
import torch
import lmp.model



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
# ============= tokenizer ==================
# train a byte-pair encoding (a list of subwords)
bpetknzr = lmp.tknzr.BPETknzr()
ds = lmp.dset.ChPoemDset()
bpetknzr.build_vocab(ds)
# build the subword tokenizer


# %%
# ============== model ===================
model = lmp.model.ElmanNet(tknzr = bpetknzr)
# initialize way 1
# torch.nn.init.zeros_(model.fc_e2h.bias)
# initialize way 2
model.params_init()
# encode mini-batch
tokenized_batches = []
MAXSEQLEN = 50
for i in range(len(ds)):
    traintext = ds[i]
    # truncate
    traintext = traintext[:MAXSEQLEN]
    padded_tkids = bpetknzr.pad_to_max(
        tkids = bpetknzr.enc(txt = traintext),
        max_seq_len = MAXSEQLEN
    )
    tokenized_batches.append(padded_tkids)
#%%
# ============= Next word predicion =============
# convert to to tensor
# Convert mini-batch to tensor.
tokenized_batches = tokenized_batches[:200]
batch_tkids = torch.LongTensor(tokenized_batches)
print(batch_tkids)
#%%
# Create language model instance.
model = lmp.model.ElmanNet(tknzr=bpetknzr)



# Calculate next token prediction.
pred, batch_cur_states = model.pred(
  batch_cur_tkids=batch_tkids,
  batch_prev_states=None,
)
# %%
print(f'vocab size: {len(bpetknzr.id2tk)}')
print(pred.shape) # (BATCH_SIZE, MAXSEQLEN, VOCAB_SIZE)


def show_prediction(exid):
    one_example = ds[exid]
    one_example_prediction = pred[0,:]
    tok_prediction = ''
    for i in range(MAXSEQLEN):
        i_predvec = one_example_prediction[i,:]
        i_predtid = torch.argmax(i_predvec).item()
        pred_tok = bpetknzr.id2tk[i_predtid]
        tok_prediction += pred_tok
    print(f'Original sentence: {one_example}')
    print(f'Prediction of NWP: {tok_prediction}')

for id in range(10):
    print(f'example {id}')
    show_prediction(id)
# # %%

# MUST TRAIN The model first
# how to use gpu?
# python -m lmp.script.train_model Elman-Net \
#   --batch_size 32 \
#   --beta1 0.9 \
#   --beta2 0.999 \
#   --ckpt_step 1000 \
#   --d_emb 100 \
#   --d_hid 100 \
#   --dset_name chinese-poem \
#   --eps 1e-8 \
#   --exp_name my_model_exp \
#   --init_lower -0.1 \
#   --init_upper 0.1 \
#   --label_smoothing 0.0 \
#   --log_step 500 \
#   --lr 1e-3 \
#   --max_norm 10 \
#   --max_seq_len 32 \
#   --n_lyr 1 \
#   --p_emb 0.5 \
#   --p_hid 0.1 \
#   --stride 32 \
#   --tknzr_exp_name ch_poem_tknzr \
#   --total_step 10000 \
#   --ver train \
#   --warmup_step 5000 \
#   --weight_decay 0.0