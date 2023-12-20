import glob
import gc
import os
from copy import copy
import pandas as pd
import numpy as np
import importlib
import sys
from tqdm import tqdm
import argparse
import torch
import transformers
import neptune
import random
from neptune.utils import stringify_unsupported
from collections import defaultdict
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset, Subset
from utils import set_seed
from decouple import config
import torch._dynamo

NEPTUNE_API_TOKEN = None
BASEDIR= '.'
for DIRNAME in 'configs data models postprocess metrics'.split():
    sys.path.append(f'{BASEDIR}/{DIRNAME}/')

parser = argparse.ArgumentParser(description="")

parser.add_argument("-C", "--config", help='config file name', type=str, default='cfg')
parser.add_argument("-G", "--gpu", help="device number for experiment", type=str, default='1')
parser.add_argument("-cmnt", "--comment", type=str, help='Neptune comment for the run', default="")

parser_args, other_args = parser.parse_known_args(sys.argv)
cfg = copy(importlib.import_module(parser_args.config).cfg)  
cfg.comment = parser_args.comment
os.environ["CUDA_VISIBLE_DEVICES"] = parser_args.gpu

#overwrite parameters from command line
if len(other_args) > 1:
    other_args = {k.replace('-',''):v for k, v in zip(other_args[1::2], other_args[2::2])}
    
    for key in other_args:
        if key in cfg.__dict__:
            
            print(f"Overwriting cfg.{key}: {cfg.__dict__[key]} -> {other_args[key]}")
            cfg_type = type(cfg.__dict__[key])
            if cfg_type == bool:
                cfg.__dict__[key] = other_args[key] == 'True'
            elif cfg_type == type(None):
                cfg.__dict__[key] = other_args[key]
            else:
                cfg.__dict__[key] = cfg_type(other_args[key])
            
cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
if cfg.seed < 0:
    cfg.seed = np.random.randint(1_000_000) 
set_seed(cfg.seed)   
print(f"Seed : {cfg.seed}")

CustomDataset = importlib.import_module(cfg.dataset).CustomDataset   
Net = importlib.import_module(cfg.model).Net
batch_to_device = importlib.import_module(cfg.dataset).batch_to_device
calc_metric = importlib.import_module(cfg.metric).calc_metric

neptune_run = None
if cfg.logging:
    # Start neptune
    fns = [parser_args.config] + [getattr(cfg, s) for s in 'dataset model metric '.split()]
    fns = sum([glob.glob(f"{BASEDIR }/*/{fn}.py") for fn in  fns], [])

    if cfg.neptune_project == "common/quickstarts":
        neptune_api_token=neptune.ANONYMOUS_API_TOKEN
    else:
        neptune_api_token=NEPTUNE_API_TOKEN
    
    neptune_run = neptune.init_run(
            project=cfg.neptune_project,
            tags="baseline",
            mode="async",
            api_token=neptune_api_token,
            capture_stdout=False,
            capture_stderr=False,
            source_files=fns,
            description=cfg.comment,
            git_ref=False
        )
    print(f"Neptune system id : {neptune_run._sys_id}")
    print(f"Neptune URL       : {neptune_run.get_url()}")
    neptune_run["cfg"] = stringify_unsupported(cfg.__dict__)

df = pd.read_parquet("datamount/train_data.parquet")
df_synthetic = pd.read_parquet("datamount/synthetic_data.parquet")
train_dataset = CustomDataset(df, cfg, mode="train", df_synthetic=df_synthetic)
val_dataset = CustomDataset(df, cfg, mode="val")
#train_dataset = CustomDataset(df, cfg, mode="test", df_synthetic=df_synthetic)

train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=cfg.batch_size,
    num_workers=cfg.num_workers,
    pin_memory=cfg.pin_memory,
    drop_last=True
)

val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_val,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=True
)
    
model = Net(cfg).to(cfg.device)
model = torch.compile(model)

total_steps = len(train_dataset)
optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
scheduler = transformers.get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=cfg.warmup * (total_steps //cfg.batch_size),
    num_training_steps=cfg.epochs * (total_steps //cfg.batch_size),
    num_cycles=0.5
)
scaler = GradScaler()

cfg.curr_step = 0
optimizer.zero_grad()
i = 0

if not os.path.exists(f"{cfg.output_dir}/fold{cfg.fold}/"): 
    os.makedirs(f"{cfg.output_dir}/fold{cfg.fold}/")
    
for epoch in range(cfg.epochs):
    cfg.curr_epoch = epoch  
    progress_bar = tqdm(range(len(train_dataloader))[:], desc=f'Train epoch {epoch}', ascii=' >=')
    tr_it = iter(train_dataloader)
    losses = []
    loss_ce = []
    losses_bpp = []
    gc.collect()
    
    model.train()
    for itr in progress_bar:
        i += 1
        cfg.curr_step += cfg.batch_size
        data = next(tr_it)
        torch.set_grad_enabled(True)
        batch = batch_to_device(data, cfg.device)
        if cfg.mixed_precision:
            with autocast():
                output_dict = model(batch)
        else:
            output_dict = model(batch)

        loss = output_dict['loss']
        losses.append(loss.item())

        if cfg.grad_accumulation >1:
            loss /= cfg.grad_accumulation
        
        if cfg.mixed_precision:
            scaler.scale(loss).backward()
         
            if i % cfg.grad_accumulation == 0:
                if cfg.clip_grad > 0:
                    scaler.unscale_(optimizer)                          
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
        else:       
            loss.backward()
            if i % cfg.grad_accumulation == 0:
                if cfg.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
                optimizer.step()
                optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
            
        if cfg.logging:
                neptune_run["train/loss"].log(value=(loss.item() * cfg.grad_accumulation), step=cfg.curr_step)
                neptune_run["lr"].log(value=optimizer.param_groups[0]['lr'], step=cfg.curr_step)
        
    print(f'Epoch {epoch} loss: \t {np.mean(losses):.5f} loss_bpp:{np.mean(losses):.5f}')
    
    if ((epoch + 1) % cfg.eval_epochs == 0 or (epoch + 1) == cfg.epochs) and not cfg.test:
        model.eval()
        torch.set_grad_enabled(False)    
        val_data = defaultdict(list)
        val_score = 0
        val_losses = []
        val_losses_ce = []
        
        for index, data in enumerate(tqdm(val_dataloader, desc=f'Val epoch {epoch}', ascii=' >=')):
            batch = batch_to_device(data, cfg.device)
            if cfg.mixed_precision:
                with autocast():
                    output_dict = model(batch)
            else:
                output_dict = model(batch)
            target = output_dict['target']
            predictions = output_dict['predictions']
            loss = output_dict['loss']
            val_losses.append(loss.item())
            if 'target' not in val_data:
                val_data['target'] = target
            else:
                val_data['target'] = torch.cat([val_data['target'], target], dim=0)

            if 'predictions' not in val_data:
                val_data['predictions'] = predictions
            else:
                val_data['predictions'] = torch.cat([val_data['predictions'], predictions], dim=0)
        
        val_score = calc_metric(cfg, val_data)
        if cfg.logging:
            neptune_run["val/loss"].log(value=val_score, step=cfg.curr_step)  
        print(f"Val combined loss:{np.mean(val_losses):.5f} Validation score: \t {val_score:.5f}") 
        
    if not cfg.save_only_last_ckpt:
        if neptune_run is not None:
            torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{cfg.fold}/{neptune_run._sys_id}_seed{cfg.seed}.pt") 
        else: 
            torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pt") 

        
if neptune_run is not None:
    torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{cfg.fold}/{neptune_run._sys_id}_seed{cfg.seed}.pt") 
else: 
    torch.save({"model": model.state_dict()}, f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pt") 
print(f"Checkpoint save: " + f"{cfg.output_dir}/fold{cfg.fold}/checkpoint_last_seed{cfg.seed}.pt")    


