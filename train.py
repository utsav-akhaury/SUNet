import os
from this import d
# from traceback import print_tb

import torch
import yaml

from utils import network_parameters
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Lambda
from torch.nn.functional import interpolate

import time
import utils
import numpy as np
import random
# from data_RGB import get_training_data, get_validation_data

from warmup_scheduler import GradualWarmupScheduler
# from tqdm import tqdm
from tensorboardX import SummaryWriter
from model.SUNet import SUNet_model

import gc
# from GPUtil import showUtilization as gpu_usage
# from numba import cuda as cd

def free_gpu_cache():                          
    gc.collect()
    torch.cuda.empty_cache()

free_gpu_cache() 

## Set Seeds
torch.backends.cudnn.benchmark = True
random.seed(1234)
np.random.seed(1234)
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)

## Load yaml configuration file
with open('training.yaml', 'r') as config:
    opt = yaml.safe_load(config)
Train = opt['TRAINING']
OPT = opt['OPTIM']
SUNet = opt['SWINUNET']

## Build Model
print('==> Build the model')
model_restored = SUNet_model(opt)
p_number = network_parameters(model_restored)
model_restored.cuda()

## Training model path direction
mode = opt['MODEL']['MODE']

model_dir = os.path.join(Train['SAVE_DIR'], mode, 'models')

## GPU
gpus = ','.join([str(i) for i in opt['GPU']])
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
device_ids = [i for i in range(torch.cuda.device_count())]
if torch.cuda.device_count() > 1:
    print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")
if len(device_ids) > 1:
    model_restored = nn.DataParallel(model_restored, device_ids=device_ids)

## Log
log_dir = os.path.join(Train['SAVE_DIR'], mode, 'log')
utils.mkdir(log_dir)
writer = SummaryWriter(log_dir=log_dir, filename_suffix=f'_{mode}')

## Optimizer
start_epoch = 1
new_lr = float(OPT['LR_INITIAL'])
optimizer = optim.Adam(model_restored.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

## Scheduler (Strategy)
warmup_epochs = 3
scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, OPT['EPOCHS'] - warmup_epochs,
                                                        eta_min=float(OPT['LR_MIN']))
scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
scheduler.step()

## Resume (Continue training by a pretrained model)
if Train['RESUME']:
    path_chk_rest = utils.get_last_path(model_dir, '_latest_ep-400_bs-16_ps-1.pth')
    utils.load_checkpoint(model_restored, path_chk_rest)
    start_epoch = utils.load_start_epoch(path_chk_rest) + 1
    utils.load_optim(optimizer, path_chk_rest)

    for i in range(1, start_epoch):
        scheduler.step()
    new_lr = scheduler.get_lr()[0]
    print('------------------------------------------------------------------')
    print("==> Resuming Training with learning rate:", new_lr)
    print('------------------------------------------------------------------')

## Loss
L1_loss = nn.L1Loss()

## DataLoaders
print('==> Loading datasets')

# Read Saved Batches   
x_train = np.load('/home/users/a/akhaury/scratch/SingleChannel_Deconv/x_train.npy')
y_train = np.load('/home/users/a/akhaury/scratch/SingleChannel_Deconv/y_train.npy')

# Normalize targets
x_train = x_train - np.mean(x_train, axis=(1,2), keepdims=True)
norm_fact = np.max(x_train, axis=(1,2), keepdims=True) 
x_train /= norm_fact

# Normalize & scale tikho inputs
y_train = y_train - np.mean(y_train, axis=(1,2), keepdims=True)
y_train /= norm_fact

# NCHW convention
y_train = np.moveaxis(y_train, -1, 1)
x_train = np.moveaxis(x_train, -1, 1)

# Convert to torch tensor
y_train = torch.tensor(y_train)
x_train = torch.tensor(x_train)
print(y_train.size(), x_train.size())

free_gpu_cache() 

## Data Augmentation funciton
def augmentation(im, seed):
    random.seed(seed)
    a = random.choice([0,1,2,3])
    if a==0:
        return im
    elif a==1:
        ch = random.choice([1, 2, 3])
        return torch.rot90(im, ch, [-2,-1])
    elif a==2:
        ch = random.choice([-2, -1])
        return torch.flip(im, [ch])
    elif a==3:
        ch1 = random.choice([10, -10])
        ch2 = random.choice([-2, -1])
        return torch.roll(im, ch1, dims=ch2)


# Show the training configuration
print(f'''==> Training details:
------------------------------------------------------------------
    Restoration mode:   {mode}
    Train patches size: {str(Train['TRAIN_PS']) + 'x' + str(Train['TRAIN_PS'])}
    Val patches size:   {str(Train['VAL_PS']) + 'x' + str(Train['VAL_PS'])}
    Model parameters:   {p_number}
    Start/End epochs:   {str(start_epoch) + '~' + str(OPT['EPOCHS'])}
    Batch sizes:        {OPT['BATCH']}
    Learning rate:      {OPT['LR_INITIAL']}
    GPU:                {'GPU' + str(device_ids)}''')
print('------------------------------------------------------------------')

# Start training!
print('==> Training start: ')
best_psnr = 0
best_ssim = 0
best_epoch_psnr = 0
best_epoch_ssim = 0
total_start_time = time.time()

for epoch in range(start_epoch, OPT['EPOCHS'] + 1):
    epoch_start_time = time.time()
    epoch_loss = 0
    train_id = 1

    # Randomly split train-validation set for every epoch
    n_obj = y_train.size()[0]
    n_train = np.int16(0.9*n_obj)

    ind = np.arange(n_obj)
    np.random.shuffle(ind)

    train_dataset = TensorDataset(x_train[ind][:n_train], y_train[ind][:n_train])
    val_dataset = TensorDataset(x_train[ind][n_train:], y_train[ind][n_train:])

    train_loader = DataLoader(dataset=train_dataset, batch_size=OPT['BATCH'],
                            shuffle=True, num_workers=0, drop_last=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=0,
                            drop_last=False)

    del train_dataset, val_dataset
    free_gpu_cache() 

    model_restored.train()
    for i, data in enumerate(train_loader, 0):
        # Forward propagation
        for param in model_restored.parameters():
            param.grad = None
        target = data[0].cuda()
        input_ = data[1].cuda()
        # print(target.size(), input_.size())
        seed = random.randint(0,1000000)
        target = Lambda(lambda x: torch.stack([augmentation(x_, seed) for x_ in x]))(target)
        input_ = Lambda(lambda x: torch.stack([augmentation(x_, seed) for x_ in x]))(input_)
        restored = model_restored(input_)
        # print(restored.size())

        if restored.size() != target.size():
            restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')

        # Compute loss
        # loss = Charbonnier_loss(restored, target)
        loss = L1_loss(restored, target)   

        del target, input_, data
        free_gpu_cache() 

        # Back propagation
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    ## Evaluation (Validation)
    if epoch % Train['VAL_AFTER_EVERY'] == 0:
        model_restored.eval()
        psnr_val_rgb = []
        ssim_val_rgb = []
        for ii, data_val in enumerate(val_loader, 0):
            target = data_val[0].cuda()
            input_ = data_val[1].cuda()
            with torch.no_grad():
                restored = model_restored(input_)
                if restored.size() != target.size():
                    restored = interpolate(restored, size=(target.size()[-2], target.size()[-1]), mode='nearest-exact')

            for res, tar in zip(restored, target):
                psnr_val_rgb.append(utils.torchPSNR(res, tar))
                ssim_val_rgb.append(utils.torchSSIM(restored, target))

            del target, input_, data_val
            free_gpu_cache() 

        psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()
        ssim_val_rgb = torch.stack(ssim_val_rgb).mean().item()

        # Save the best PSNR model of validation
        if psnr_val_rgb > best_psnr:
            best_psnr = psnr_val_rgb
            best_epoch_psnr = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestPSNR_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE']))) 
        print("[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (
            epoch, psnr_val_rgb, best_epoch_psnr, best_psnr))

        # Save the best SSIM model of validation
        if ssim_val_rgb > best_ssim:
            best_ssim = ssim_val_rgb
            best_epoch_ssim = epoch
            torch.save({'epoch': epoch,
                        'state_dict': model_restored.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, "model_bestSSIM_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE'])))
        print("[epoch %d SSIM: %.4f --- best_epoch %d Best_SSIM %.4f]" % (
            epoch, ssim_val_rgb, best_epoch_ssim, best_ssim))

        """ 
        # Save evey epochs of model
        torch.save({'epoch': epoch,
                    'state_dict': model_restored.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))
        """

        writer.add_scalar('val/PSNR', psnr_val_rgb, epoch)
        writer.add_scalar('val/SSIM', ssim_val_rgb, epoch)
    scheduler.step()

    print("------------------------------------------------------------------")
    print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tLearningRate {:.6f}".format(epoch, time.time() - epoch_start_time,
                                                                              epoch_loss, scheduler.get_lr()[0]))
    print("------------------------------------------------------------------")

    # Save the last model
    torch.save({'epoch': epoch,
                'state_dict': model_restored.state_dict(),
                'optimizer': optimizer.state_dict()
                }, os.path.join(model_dir, "model_latest_ep-{}_bs-{}_ps-{}.pth".format(OPT['EPOCHS'], OPT['BATCH'], SUNet['PATCH_SIZE'])))

    writer.add_scalar('train/loss', epoch_loss, epoch)
    writer.add_scalar('train/lr', scheduler.get_lr()[0], epoch)

    del train_loader, val_loader, epoch_loss
    free_gpu_cache() 

writer.close()

total_finish_time = (time.time() - total_start_time)  # seconds
print('Total training time: {:.1f} seconds'.format((total_finish_time)))
