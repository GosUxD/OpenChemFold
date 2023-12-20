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
cfg.train_val = False
cfg.eval_epochs = 1
cfg.seed = 1994

#logging
cfg.neptune_project = "common/quickstarts"
cfg.neptune_connection_mode = "async"
cfg.tags = "base"
cfg.comment = ""
cfg.logging = True

#dataset
cfg.dataset = "ds_1"
cfg.max_sequence = 206

#model
cfg.model = "mdl_1_squeezeformer"
cfg.d_model = 192

encoder_config = SimpleNamespace(**{})
encoder_config.input_dim=192
encoder_config.encoder_dim=192
encoder_config.num_layers=14
encoder_config.num_attention_heads= 6
encoder_config.feed_forward_expansion_factor=4
encoder_config.conv_expansion_factor= 2
encoder_config.input_dropout_p= 0.1
encoder_config.feed_forward_dropout_p= 0.1
encoder_config.attention_dropout_p= 0.1
encoder_config.conv_dropout_p= 0.1
encoder_config.conv_kernel_size= 51

cfg.encoder_config = encoder_config

#optimization & scheduler
cfg.fold = 0
cfg.epochs = 200
cfg.lr = 5e-4
cfg.optimizer = "AdamW"
cfg.weight_decay = 0.05
cfg.clip_grad = 0.
cfg.track_norm = False
cfg.warmup = 0.5
cfg.batch_size = 64
cfg.batch_size_val = 256
cfg.pin_memory = False
cfg.grad_accumulation = 1
cfg.num_workers = 4
cfg.mixed_precision = True #False

#postprocess
cfg.metric = "metric"

#eval
cfg.eval_epochs = 1
cfg.save_val_data = False

#saving
cfg.save_only_last_ckpt = False