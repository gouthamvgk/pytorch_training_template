import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
import argparse
import torch.distributed as dist
import yaml
import time
from pathlib import Path
import torch.optim as optim
import sys
import numpy as np
from models.sample_model import my_model
from utils.common import increment_path, init_seeds, clean_checkpoint, reduce_tensor, time_synchronized, ModelEMA, loss_fn, test_model
from utils.dataset import my_dataset
from torch.utils.tensorboard import SummaryWriter

def change_lr(epoch, config, optimizer):
    if epoch >= config['optimizer_params']['step_epoch']:
        curr_lr = config['optimizer_params']['lr']
        changed_lr = curr_lr * (config['optimizer_params']['step_value'] ** (epoch-config['optimizer_params']['step_epoch']))
    else:
        changed_lr = config['optimizer_params']['lr']
    for g in optimizer.param_groups:
        g['lr'] = changed_lr
        c_lr = g['lr']
        print("Changed learning rate to {}".format(c_lr))

def train(config, rank):
    is_distributed = (rank >=0)
    save_dir = Path(config['train_params']['save_dir'])
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)
    results_file = None 
    if rank in [0, -1]: results_file = open(save_dir / "results.txt", 'a')
    with open(save_dir / 'config.yaml', 'w') as file:
        yaml.dump(config, file, sort_keys=False)
    init_seeds(rank + config['train_params']['init_seed'])
    model = my_model(config['model_params']).to(device)
    start_epoch = config['train_params']['start_epoch'] if config['train_params']['start_epoch'] > -1 else 0
    if config['model_params']['restore_path']:
        restore_dict = torch.load(config['model_params']['restore_path'], map_location=device)
        model.load_state_dict(clean_checkpoint(restore_dict['model'] if 'model' in restore_dict else restore_dict))
        print("Restored model weights..")
        if config['train_params']['start_epoch'] < 0:
            start_epoch = restore_dict['epoch'] + 1
            print("Set epoch number as {}".format(start_epoch))
    pg0, pg1, pg2 = [], [], []
    for k, v in model.named_modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)  # biases
        if isinstance(v, nn.BatchNorm2d) or isinstance(v, nn.BatchNorm1d):
            pg0.append(v.weight)  # no decay
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)  # apply decay
        else:
            print("Module {} doesn't belong to given param groups, so ignoring it!!!. Add it manually if required")
    if config['optimizer_params']['opt_type'].lower() == "adam":
        optimizer = optim.Adam(pg0, lr=config['optimizer_params']['lr'], betas=(0.9, 0.999))  # adjust beta1 to momentum
    else:
        optimizer = optim.SGD(pg0, lr=config['optimizer_params']['lr'], momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': pg1, 'weight_decay': config['optimizer_params']['weight_decay']})
    optimizer.add_param_group({'params': pg2}) 
    print('Optimizer groups: %g .bias, %g conv.weight, %g other' % (len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2
    if config['model_params']['restore_path']:
        if ('optimizer' in restore_dict) and config['train_params']['restore_opt']:
            optimizer.load_state_dict(restore_dict['optimizer'])
            print("Restored optimizer...")
    ema = None
    if config['train_params']['use_ema']:
        ema = ModelEMA(model) if rank in [-1, 0] else None
        print("Keeping track of weights in ema..")
        if config['model_params']['restore_path']:
            if ('ema' in restore_dict):
                print("Restoring ema weights from the given ckpt")
                ema.ema.load_state_dict(restore_dict['ema'])
                ema.updates = restore_dict['ema_updates']
    if is_distributed and config['train_params']['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(device)
    if is_distributed:
        model = DDP(model, device_ids=[rank], output_device=rank)
    train_dataset = my_dataset(config['dataset_params'], typ="train")
    sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config['train_params']['batch_size'],
                                            num_workers=config['train_params']['num_workers'],
                                            shuffle = False if is_distributed else True,
                                            sampler=sampler,
                                            collate_fn=None,
                                            pin_memory=True)
    num_batches = len(train_dataloader)
    if rank in [-1, 0]:
        val_dataset = my_dataset(config['dataset_params'], typ="val")
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1,
                                            num_workers=0,
                                            sampler=None,
                                            collate_fn=None,
                                            pin_memory=True)
    start_time = time.time()
    num_epochs = config['train_params']['num_epochs']
    best_val_score = 1e-10
    if rank in [-1, 0]: print("Started training for {} epochs".format(num_epochs))
    print("Number of batches: {}".format(num_batches))
    warmup_iters = config['optimizer_params']['warmup_epochs'] * num_batches
    change_lr(start_epoch, config, optimizer)
    loss = loss_fn(config['loss_params'])
    for epoch in range(start_epoch, num_epochs):
        print("Started epoch: {} in rank {}".format(epoch + 1, rank))
        model.train()
        if rank != -1:
            train_dataloader.sampler.set_epoch(epoch)
        pbar = enumerate(train_dataloader)
        if rank in [-1, 0]:
            pbar = tqdm(pbar, total=num_batches)
        optimizer.zero_grad()
        mloss = torch.zeros(3, device=device)
        if rank in [-1, 0]: print(('\n' + '%10s' * 6) % ('Epoch', 'gpu_mem', 'Iteration','TotLoss', 'Dtime' , 'Mtime'))
        t5 = time_synchronized()
        for i, (ins_data, gt) in pbar:
            ni = i + num_batches * epoch
            if ni < warmup_iters:
                xi = [0, warmup_iters]
                for _, x in enumerate(optimizer.param_groups):
                    x['lr'] = np.interp(ni, xi, [0.0, config['optimizer_params']['lr']])
            t1 = time_synchronized()
            ins_data = ins_data.to(device, non_blocking=True)
            gt_data = gt_data.to(device, non_blocking=True)
            model_out = model(ins_data)
            total_loss = loss(model_out, gt_data)
            total_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if ema:
                ema.update(model)
            t4 = time_synchronized()
            data_time, model_time = torch.tensor(t1 - t5, device=device), torch.tensor(t4-t1, device=device)
            loss_items  = torch.stack((total_loss, data_time, model_time)).detach()
            if is_distributed: loss_items = reduce_tensor(loss_items)
            if rank in [-1, 0]:
                mloss = (mloss * i + loss_items) / (i + 1)
                mem = '%.3gG' % (torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0)  # (GB)
                s = ('%10s' * 2 + '%10.4g' * 4) % (str(epoch),mem, i, *mloss)
                pbar.set_description(s)
                if ((i+1) % config['train_params']['log_interval']) == 0:
                    write_str = "Epoch: {} Iter: {}, Loss: {}\n".format(epoch, i, mloss[0].item())
                    results_file.write(write_str)
                if ((i+1) % 2000) == 0:
                    ckpt = {'epoch': epoch,
                            'iter': i,
                            'ema': ema.ema.state_dict() if ema else None,
                            'ema_updates': ema.updates if ema else 0,
                            'model': model.module.state_dict() if is_distributed else model.state_dict(),
                            'optimizer': optimizer.state_dict()}
                    torch.save(ckpt, weight_dir / 'lastiter.pt')
                    if use_wandb:
                        wandb.save(str(weight_dir / 'lastiter.pt'))
                t5 = time_synchronized()
        if rank in [-1, 0]:
            print("\nDoing evaluation..")
            with torch.no_grad():
                if ema:
                    eval_model = ema.ema
                else:
                    eval_model = model.module if is_distributed else model
                results = test_model(val_dataloader, eval_model, device)
            ckpt = {'epoch': epoch,
                    'iter': -1,
                    'ema': ema.ema.state_dict() if ema else None,
                    'ema_updates': ema.updates if ema else 0,
                    'model': model.module.state_dict() if is_distributed else model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'metrics': results}
            torch.save(ckpt, weight_dir / 'last.pt')
            if use_wandb:
                wandb.save(str(weight_dir / 'last.pt'))
                results_file.flush()
                wandb.save(str(save_dir / "results.txt"))
            if results['metric'] > best_val_score:
                best_val_score = results['metric']
                print("Saving best model at epoch {} with score {}".format(epoch, best_val_score))
                torch.save(ckpt, weight_dir / 'best.pt')
                if use_wandb:
                    wandb.save(str(weight_dir / 'best.pt'))
        change_lr(epoch, config, optimizer)
    if rank > 0:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default="configs/sample_config.yaml", help="Path to the config file")
    parser.add_argument('--local_rank', type=int, default=-1, help="Rank of the process incase of DDP")
    opt = parser.parse_args()
    opt.world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if opt.local_rank >=0:
        assert torch.cuda.device_count() > opt.local_rank
        torch.cuda.set_device(opt.local_rank)
        device = torch.device('cuda', opt.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
    else:
        if "cpu" not in device: torch.cuda.set_device(device)
    with open(opt.config_path, 'r') as file:
        config = yaml.full_load(file)
    config["train_params"]['save_dir'] = increment_path(Path(config['train_params']['output_dir']) / config['train_params']['experiment_name'])
    if opt.local_rank in [0, -1]:
        for i,k in config.items():
            print("{}: ".format(i))
            print(k)
    use_wandb = False
    if config['train_params']['use_wandb']:
        import wandb
        wandb.init(name=config['train_params']['experiment_tag'], config=config, notes="train", project="my_project")
        use_wandb = True
    train(config, opt.local_rank)
