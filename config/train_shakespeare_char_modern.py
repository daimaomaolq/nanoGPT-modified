# train a miniature character-level Shakespeare model with modern LLM components
# compared to train_shakespeare_char.py, this swaps in RMSNorm, SwiGLU, and RoPE

out_dir = 'out-shakespeare-char-modern'
eval_interval = 250
eval_iters = 200
log_interval = 10

always_save_checkpoint = False

wandb_log = False
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-gpt-modern'

dataset = 'shakespeare_char'
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256

n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
bias = False

norm_type = 'rmsnorm'
mlp_type = 'swiglu'
position_embedding_type = 'rope'
rope_base = 10000.0
swiglu_hidden_mult = 8/3

learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99

warmup_iters = 100

# on CPU/Windows also add:
# device = 'cpu'
# compile = False
