import datetime 
import json 
import os 

import numpy as np 
import torch 
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim 
from torch.utils.data import DataLoader

from model.clip import CLIP
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import ClipDataset, dataset_collate
from utils.utils import (get_configs, get_lr_scheduler, set_optimizer_lr,
                         show_config)
from utils.utils_fit import fit_one_epoch
from model.simple_tokenizer import SimpleTokenizer


if __name__ == "__main__":
    Cuda = True 
    distributed = False
    fp16 = False
    model_path = "ViT-B-32-OpenAI.pth"
    phi = "openai/VIT-B-32"

    batch_size = 16
    init_epoch = 0
    epoch = 100

    init_lr = 1e-4
    min_lr = init_lr * 0.01

    optimizer_type = "adamw"
    momentum = 0.9
    weight_decay = 1e-2

    lr_decay_type = "cos" 

    save_period = 1
    save_dir = "logs"

    eval_flag = True 
    eval_period = 1

    num_workers = 4

    datasets_path = "datasets/"
    datasets_train_json_path = "datasets/en_train.json"
    datasets_val_json_path = "datasets/en_val.json"
    datasets_random = True

    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0
        rank            = 0

    config  = get_configs(phi)
    model   = CLIP(**config)
    # 新添加的代码 #
    _tokenizer = SimpleTokenizer()
    # ----------- #

    if model_path != '':

        if local_rank == 0:
            print(f'Load weights {model_path}.')

        model_dict      = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location = device)
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)

        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
            print("\n\033[1;33;44m 注意, head部分没有载入是正常现象, Backbone部分没有载入是错误的。\033[0m")  # It is normal for the head part to be not loaded, and it is wrong for the backbone part to be not loaded

    'visual.proj' in model_dict.keys()

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y_%m_%d_%H_%M_%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, None)
    else:
        loss_history = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler # ???
        scaler = GradScaler()
    else:
        scaler = None  

    model_train = model.train()
    if Cuda:
        if distributed:
            #----------------------------#
            #   多卡平行运行   Multi-card parallel operation
            #----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True
            model_train = model_train.cuda()


    train_lines = json.load(open(datasets_train_json_path, mode = 'r', encoding = 'utf-8'))
    val_lines   = json.load(open(datasets_val_json_path, mode = 'r', encoding = 'utf-8'))
    num_train   = len(train_lines)
    num_val     = len(val_lines)

    if local_rank == 0:
        show_config(
            model_path = model_path, phi = phi, \
            Init_Epoch = init_epoch, Epoch = epoch, batch_size = batch_size, \
            Init_lr = init_lr, Min_lr = min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val
        )


    if True:
        nbs             = 64
        lr_limit_max    = 1e-4
        lr_limit_min    = 3e-5
        init_lr_fit     = min(max(batch_size / nbs * init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit      = min(max(batch_size / nbs * min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adamw' : optim.AdamW(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'adam'  : optim.Adam(model.parameters(), init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
        }[optimizer_type]

        lr_scheduler_func = get_lr_scheduler(lr_decay_type, init_lr_fit, min_lr_fit, epoch)

        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")  # The dataset is too small to continue training, please expand the dataset.
        
        train_dataset   = ClipDataset([config['input_resolution'], config['input_resolution']], train_lines, datasets_path, random = datasets_random)
        val_dataset     = ClipDataset([config['input_resolution'], config['input_resolution']], val_lines, datasets_path, random = False)

        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        gen_val         = DataLoader(val_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                drop_last=False, collate_fn=dataset_collate, sampler=val_sampler)
        
        #----------------------#
        #   记录eval的map曲线  Record the map curve of eval
        #----------------------# 
        if local_rank == 0:
            eval_dataset    = ClipDataset([config['input_resolution'], config['input_resolution']], val_lines, datasets_path, random = False)
            gen_eval        = DataLoader(eval_dataset, shuffle=False, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                                    drop_last=False, collate_fn=dataset_collate, sampler=None)
            eval_callback   = EvalCallback(model, gen_eval, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None

        for epoch in range(init_epoch, epoch):
            if distributed:
                train_sampler.set_epoch(epoch)
            
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, epoch, Cuda, \
                            fp16, scaler, save_period, save_dir, local_rank, _tokenizer)
            
        if local_rank == 0:
            loss_history.writer.close()