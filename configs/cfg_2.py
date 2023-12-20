import os
from types import SimpleNamespace

cfg = SimpleNamespace(**{})
cfg.debug = True

#paths
cfg.name = os.path.basename(__file__).split('.')[0]
cfg.output_dir = f"datamount/weights/{os.path.basename(__file__).split('.')[0]}"
cfg.data_folder = "datamount/"
cfg.train_df = "datamount/train_data.parquet"

#stages
cfg.test = False
cfg.train = True
cfg.seed = 1994

#logging
cfg.neptune_project = "common/quickstarts"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"
cfg.comment = ""
cfg.logging = True

#dataset
cfg.dataset = "ds_2"
cfg.max_sequence = 206


#optimization & scheduler
cfg.fold = 0
cfg.epochs = 60
cfg.lr = 1e-3
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.05
cfg.clip_grad = 4.
cfg.track_norm = False
cfg.warmup = 0.5
cfg.batch_size = 8
cfg.batch_size_val = 8
cfg.pin_memory = False
cfg.grad_accumulation = 4.
cfg.num_workers = 4
cfg.mixed_precision = True #False

#postprocess
cfg.metric = "metric"

#eval
cfg.eval_epochs = 1
cfg.save_val_data = False

#saving
cfg.save_only_last_ckpt = False

#MODEL
cfg.model = "mdl_2_twintower"
cfg.msa_depth = 64
cfg.d_model = 128
cfg.ce_ignore_index = -100
cfg.padding_index = 0
cfg.vocab_size = 5


#MSA EMBEDDER CONFIG
msa_embedder = SimpleNamespace(**{})
msa_embedder.c_m = 128
msa_embedder.c_z = 64
msa_embedder.rna_fm = True

cfg.msa_embedder = msa_embedder

#CHEMFORMER STACK CONFIG
chemformer_stack = SimpleNamespace(**{})

chemformer_stack.blocks_per_ckpt =  1,
chemformer_stack.c_m = 128
chemformer_stack.c_z = 64
chemformer_stack.c_hidden_msa_att = 32
chemformer_stack.c_hidden_opm = 32
chemformer_stack.c_hidden_mul = 64
chemformer_stack.c_hidden_pair_att = 32
chemformer_stack.c_s = 384
chemformer_stack.no_heads_msa = 8
chemformer_stack.no_heads_pair = 8
chemformer_stack.no_blocks = 8
chemformer_stack.transition_n = 4

cfg.chemformer_stack = chemformer_stack


#BPP HEAD CONFIG
bpp_head = SimpleNamespace(**{})
bpp_head.c_in = 64

cfg.bpp_head = bpp_head

#SECONDARY STRUCTURE HEAD CONFIG
ss_head = SimpleNamespace(**{})
ss_head.c_in = 384
ss_head.c_hidden = 384
cfg.ss_head = ss_head

#PREDICTED LOCAL DISTANCE DIFFERENCE TEST CONFIG
plddt_head = SimpleNamespace(**{})
plddt_head.c_in = 384
plddt_head.no_bins = 50
cfg.plddt_head = plddt_head