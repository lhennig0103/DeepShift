import argparse
import os
import random
import shutil
import time
import warnings
import sys
import csv
import distutils
import distutils.util
from contextlib import redirect_stdout
from collections import OrderedDict
import copy
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Configuration, Categorical, Integer, Float, InCondition
from ConfigSpace.conditions import AbstractCondition
from smac.multi_objective.parego import ParEGO


import torchsummary
import optim
import numpy as np

from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from unoptimized.convert import convert_to_unoptimized
import unoptimized

import matplotlib.pyplot as plt
from typing import List
import multiprocessing as mp

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float

import caltech101_models as models
from codecarbon import EmissionsTracker
from codecarbon import OfflineEmissionsTracker

from torchvision.datasets import Caltech101
from torchvision.transforms import ToTensor
from torch.utils.data import random_split
import os

'''
Unfortunately, none of the pytorch repositories with ResNets on CIFAR10 provides an 
implementation as described in the original paper. If you just use the torchvision's 
models on CIFAR10 you'll get the model that differs in number of layers and parameters. 
This is unacceptable if you want to directly compare ResNet-s on CIFAR10 with the 
original paper. The purpose of resnet_cifar10 (which has been obtained from https://github.com/akamaster/pytorch_resnet_cifar10
is to provide a valid pytorch implementation of ResNet-s for CIFAR10 as described in the original paper. 
'''
# Set the start method for multiprocessing
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CALTECH101 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

best_acc1 = 0
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_train_epochs = 20

if args.arch == 'googlenet':
    num_layers = 22
elif args.arch == 'resnet20':
    num_layers = 20
elif args.arch == 'mobilenetv2':
    num_layers = 53
else:
    raise ValueError(f"Unknown architecture: {args.arch}")

# def main(cfg, fixed_params):
#     global best_acc1

#     if not fixed_params['evaluate'] and fixed_params['use_kernel']:
#         raise ValueError('Our custom kernel currently supports inference only, not training.')


#     # Fixed parameter setup
#     if fixed_params['seed'] is not None:
#         random.seed(fixed_params['seed'])
#         torch.manual_seed(fixed_params['seed'])
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     if fixed_params['gpu'] is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')

#     if fixed_params['dist_url'] == "env://" and fixed_params['world_size'] == -1:
#         fixed_params['world_size'] = int(os.environ["WORLD_SIZE"])

#     fixed_params['distributed'] = fixed_params['world_size'] > 1 or fixed_params['multiprocessing_distributed']

#     ngpus_per_node = torch.cuda.device_count()
#     if fixed_params['multiprocessing_distributed']:
#         # Since we have ngpus_per_node processes per node, the total world_size
#         # needs to be adjusted accordingly
#         fixed_params['world_size'] = ngpus_per_node * fixed_params['world_size']
#         # Use torch.multiprocessing.spawn to launch distributed processes: the
#         # main_worker process function
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, fixed_params))
#     else:
#         # Simply call main_worker function
#         main_worker(fixed_params['gpu'], ngpus_per_node, cfg, fixed_params)

#     cs = create_configspace()
#     facades: list[AbstractFacade] = []
#     scenario = Scenario(
#             cs,
#             trial_walltime_limit=3000,  # After 60 seconds, we stop the hyperparameter optimization
#             n_trials=50,  # Evaluate max 500 different trials
#             min_budget=1,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
#             max_budget=25,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
#             n_workers=1,
#         )
    
#     # Create our intensifier
#     intensifier_object = Hyperband
#     intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")
    
#     # We want to run five random configurations before starting the optimization.
#     initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

#     smac = MFFacade(
#             scenario,
#             train_model,
#             initial_design=initial_design,
#             intensifier=intensifier,
#             overwrite=True,
#         )
#     incumbent = smac.optimize()

#     # Get cost of default configuration
#     default_cost = smac.validate(cs.get_default_configuration())
#     print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

#     # Let's calculate the cost of the incumbent
#     incumbent_cost = smac.validate(incumbent)
#     print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

#     facades.append(smac)
    
#     print(incumbent, smac.validate)


def main_worker(gpu, ngpus_per_node, cfg, fixed_params):
    global best_acc1

    # if gpu is not None:
    #     torch.cuda.set_device(gpu)
    #     model = model.cuda(gpu)
    # else:
    #     model = torch.nn.DataParallel(model).cuda()


    # # Initialize distributed training if required
    # if fixed_params['distributed']:
    #     if fixed_params['dist_url'] == "env://" and fixed_params['rank'] == -1:
    #         fixed_params['rank'] = int(os.environ["RANK"])
    #     if fixed_params['multiprocessing_distributed']:
    #         # For multiprocessing distributed training, rank needs to be the
    #         # global rank among all the processes
    #         fixed_params['rank'] = fixed_params['rank'] * ngpus_per_node + gpu
    #     dist.init_process_group(backend=fixed_params['dist_backend'], init_method=fixed_params['dist_url'],
    #                             world_size=fixed_params['world_size'], rank=fixed_params['rank'])

    # Create model
    if 'model' in fixed_params and cfg['model']:  # If a model path is specified in cfg
        if args.arch or cfg.get('pretrained'):
            print("WARNING: Ignoring 'arch' and 'pretrained' arguments when loading model from path...")
        model = None
        saved_checkpoint = torch.load(cfg['model'])
        if isinstance(saved_checkpoint, nn.Module):
            model = saved_checkpoint
        elif "model" in saved_checkpoint:
            model = saved_checkpoint["model"]
        else:
            raise Exception("Unable to load model from " + cfg['model'])

        model = model.cuda(gpu) if gpu is not None else torch.nn.DataParallel(model).cuda()
    elif fixed_params['pretrained']:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model = model.cuda(gpu) if gpu is not None else torch.nn.DataParallel(model).cuda()
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        model = model.cuda(gpu) if gpu is not None else torch.nn.DataParallel(model).cuda()

    model_rounded = None
    
    if fixed_params['freeze'] and fixed_params['pretrained']:
        for param in model.parameters():
            param.requires_grad = False

    if 'weights' in fixed_params and fixed_params['weights']:
        saved_weights = torch.load(fixed_params['weights'])
        if isinstance(saved_weights, nn.Module):
            state_dict = saved_weights.state_dict()
        elif "state_dict" in saved_weights:
            state_dict = saved_weights["state_dict"]
        else:
            state_dict = saved_weights
            
        try:
            model.load_state_dict(state_dict)
        except:
            # create new OrderedDict that does not contain module.
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove `module.`
                new_state_dict[name] = v
                
            model.load_state_dict(new_state_dict)

    if cfg['shift_depth'] > 0:
        model, _ = convert_to_shift(model, cfg['shift_depth'], cfg['shift_type'], 
                                    convert_weights=(fixed_params['pretrained'] or fixed_params['weights']), 
                                    use_kernel=fixed_params.get('use_kernel', False), 
                                    rounding=cfg.get('rounding', 'deterministic'), 
                                    weight_bits=cfg.get('weight_bits', 5), 
                                    act_integer_bits=int(cfg.get('activation_integer_bits', 16)), 
                                    act_fraction_bits=cfg.get('activation_fraction_bits', 16))
    elif fixed_params['use_kernel']==False and cfg.get('shift_depth', 0) == 0:
        model = convert_to_unoptimized(model)


    if fixed_params['distributed']:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if gpu is not None:
            torch.cuda.set_device(gpu)
            model.cuda(gpu)
            # Adjusting batch size and workers based on the number of GPUs
            cfg['batch_size'] = int(cfg['batch_size'] / ngpus_per_node)
            fixed_params['workers'] = int(fixed_params['workers'] / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
    elif gpu is not None:
        torch.cuda.set_device(gpu)
        model = model.cuda(gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # Define loss function (criterion)
    criterion = nn.CrossEntropyLoss().cuda(gpu if gpu is not None else 0)

    # Create optimizer
    model_other_params = []
    model_sign_params = []
    model_shift_params = []

    for name, param in model.named_parameters():
        if name.endswith(".sign"):
            model_sign_params.append(param)
        elif name.endswith(".shift"):
            model_shift_params.append(param)
        else:
            model_other_params.append(param)

    lr_sign = cfg['lr']
    params_dict = [
        {"params": model_other_params},
        {"params": model_sign_params, 'lr': cfg['lr'], 'weight_decay': 0},
        {"params": model_shift_params, 'lr': cfg['lr'], 'weight_decay': 0}
    ]

    # Define optimizer based on cfg
    optimizer = None
    optimizer_type = cfg['optimizer'].lower()
    if optimizer_type == "sgd":
        optimizer = torch.optim.SGD(params_dict, cfg['lr'], momentum=cfg['momentum'], weight_decay=cfg['weight_decay'])
    elif optimizer_type == "adadelta":
        optimizer = torch.optim.Adadelta(params_dict, cfg['lr'], weight_decay=cfg['weight_decay'])
    elif optimizer_type == "adagrad":
        optimizer = torch.optim.Adagrad(params_dict, cfg['lr'], weight_decay=cfg['weight_decay'])
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(params_dict, cfg['lr'], weight_decay=cfg['weight_decay'])
    elif optimizer_type == "rmsprop":
        optimizer = torch.optim.RMSprop(params_dict, cfg['lr'], weight_decay=cfg['weight_decay'])
    elif optimizer_type == "radam":
        optimizer = optim.RAdam(params_dict, cfg['lr'], weight_decay=cfg['weight_decay'])
    elif optimizer_type == "ranger":
        optimizer = optim.Ranger(params_dict, cfg['lr'], weight_decay=cfg['weight_decay'])
    else:
        raise ValueError("Optimizer type: ", optimizer_type, " is not supported or known")


    #     # Define learning rate schedule
    # if cfg.get('lr_schedule', False):
    #     if 'lr_step_size' in cfg and cfg['lr_step_size'] is not None:
    #         lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg['lr_step_size'])
    #     else:
    #         lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                             milestones=[80, 120, 160, 180], last_epoch=cfg.get('start_epoch', 0) - 1)

    if args.arch in ['resnet1202', 'resnet110']:
        # Warm-up setting for specific architectures
        for param_group in optimizer.param_groups:
            param_group['lr'] = cfg['lr'] * 0.1

    # Optionally resume from a checkpoint
    if fixed_params.get('resume') and os.path.isfile(fixed_params['resume']):
        print("=> loading checkpoint '{}'".format(fixed_params['resume']))
        checkpoint = torch.load(fixed_params['resume'])
        fixed_params['start_epoch'] = checkpoint['epoch']  # Assuming start_epoch is still managed by cfg
        best_acc1 = checkpoint['best_acc1']
        if gpu is not None:
            best_acc1 = best_acc1.to(gpu)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
            .format(fixed_params['resume'], checkpoint['epoch']))
    else:
        if fixed_params.get('resume'):
            print("=> no checkpoint found at '{}'".format(fixed_params['resume']))

    # If evaluating, round weights to ensure that the results are due to powers of 2 weights
    if fixed_params.get('evaluate', False):
        model = round_shift_weights(model)


    cudnn.benchmark = True

    model_summary = None

    try:
        model_summary, model_params_info = torchsummary.summary_string(model, input_size=(3, 32, 32))
        print(model_summary)
        print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
    except:
        print("WARNING: Unable to obtain summary of model")

    # Naming model sub-directory
    conv2d_layers_count = count_layer_type(model, nn.Conv2d) + count_layer_type(model, unoptimized.UnoptimizedConv2d)
    linear_layers_count = count_layer_type(model, nn.Linear) + count_layer_type(model, unoptimized.UnoptimizedLinear)
    
    shift_label = "shift" if cfg['shift_depth'] == 0 else "shift_ps" if cfg['shift_type'] == 'PS' else "shift_q"
    shift_label += "_all" if conv2d_layers_count == 0 and linear_layers_count == 0 else "_%s" % cfg['shift_depth']
    shift_label += "_wb_%s" % cfg['weight_bits'] if cfg['shift_depth'] > 0 else ""

    # 'desc_label = "_%s" % cfg['desc'] if cfg.get('desc') else ""'

    model_name = '%s/%s' % (args.arch, shift_label)

    if fixed_params['save_model'] == False:
        model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "models"), "caltech101"), model_name)
        os.makedirs(model_dir, exist_ok=True)

        with open(os.path.join(model_dir, 'command_args.txt'), 'w') as command_args_file:
            for arg in cfg:
                value = cfg[arg]
                command_args_file.write(arg + ": " + str(value) + "\n")


        with open(os.path.join(model_dir, 'model_summary.txt'), 'w') as summary_file:
            with redirect_stdout(summary_file):
                if model_summary is not None:
                    print(model_summary)
                    print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
                else:
                    print("WARNING: Unable to obtain summary of model")

    # Data loading code
    # data_dir = "~/pytorch_datasets"
    NUM_STREAM = 2 ###CHANGE (REDUCE) VALUE IF ERROR###

    #Change Grayscale Image to RGB for the shape
    class GrayscaleToRGB(object):
        def __call__(self, img):
            if img.mode == 'L':
                img = img.convert("RGB")
            return img

    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        GrayscaleToRGB(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])]
        )


    train_dataset = torchvision.datasets.Caltech101(root='./data', transform=transform, download=True)


    # Split the dataset into training and testing sets (test is 20%)

    TRAIN_SIZE = 0.8

    train_size = int(TRAIN_SIZE* len(train_dataset))
    test_size = len(train_dataset) - train_size
    train_dataset, test_dataset = random_split(train_dataset, [train_size, test_size])

    ## Bigger the GPU memory, the bigger the batch_size
    BATCH = 16
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH,
                                                shuffle=True)
    val_loader = torch.utils.data.DataLoader(test_dataset,batch_size=BATCH,
                                                shuffle=False)#shuffle is fixed to evaluate the model's performance on the same set of samples every time


    # if fixed_params['evaluate']:
    #     val_log = validate(val_loader, model, criterion, cfg)
    #     val_log = [val_log]

    #     with open(os.path.join(model_dir, "test_log.csv"), "w") as test_log_file:
    #         test_log_csv = csv.writer(test_log_file)
    #         test_log_csv.writerow(['test_loss', 'test_top1_acc', 'test_time'])
    #         test_log_csv.writerows(val_log)
    # else:
    train_log = []

        # with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
        #     train_log_csv = csv.writer(train_log_file)
        #     train_log_csv.writerow(['epoch', 'train_loss', 'train_top1_acc', 'train_time', 'test_loss', 'test_top1_acc', 'test_time'])

    # for epoch in range(fixed_params['start_epoch'], cfg['epochs']):
    #     if fixed_params['distributed']:
    #         train_sampler.set_epoch(epoch)

    #     # if cfg.get('alternate_update'):
    #     #     if epoch % 2 == 1:
    #     #         optimizer.param_groups[1]['lr'] = 0
    #     #         optimizer.param_groups[2]['lr'] = optimizer.param_groups[0]['lr']
    #     #     else:
    #     #         optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr']
    #     #         optimizer.param_groups[2]['lr'] = 0

    #     # Train for one epoch
    #     # print("current lr ", [param['lr'] for param in optimizer.param_groups])
    #     train_epoch_log = train(train_loader, model, criterion, optimizer, epoch, cfg, fixed_params)
    #     # if cfg.get('lr_schedule'):
    #     #     lr_scheduler.step()

    #     # Evaluate on validation set
    #     val_epoch_log = validate(val_loader, model, criterion, cfg, fixed_params)
    #     acc1 = val_epoch_log[2]

    #     # Append to log
    #     with open(os.path.join(model_dir, "train_log.csv"), "a") as train_log_file:
    #         train_log_csv = csv.writer(train_log_file)
    #         train_log_csv.writerow(((epoch,) + train_epoch_log + val_epoch_log)) 

    #     # Remember best acc@1 and save checkpoint
    #     is_best = acc1 > best_acc1
    #     best_acc1 = max(acc1, best_acc1)

    #     if fixed_params.get('print_weights'):
    #         with open(os.path.join(model_dir, 'weights_log_' + str(epoch) + '.txt'), 'w') as weights_log_file:
    #             with redirect_stdout(weights_log_file):
    #                 print("Model's state_dict:")
    #                 for param_tensor in model.state_dict():
    #                     print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    #                     print(model.state_dict()[param_tensor])

    #     if not fixed_params['multiprocessing_distributed'] or (fixed_params['multiprocessing_distributed'] and fixed_params['rank'] % ngpus_per_node == 0):
    #         if is_best:
    #             try:
    #                 if cfg.get('save_model'):
    #                     model_rounded = round_shift_weights(model, clone=True)
    #                     torch.save(model_rounded.state_dict(), os.path.join(model_dir, "weights.pth"))
    #                     torch.save(model_rounded, os.path.join(model_dir, "model.pth"))
    #             except: 
    #                 print("WARNING: Unable to save model.pth")

    #         save_checkpoint({
    #             'epoch': epoch + 1,
    #             'arch': args.arch,
    #             'state_dict': model.state_dict(),
    #             'best_acc1': best_acc1,
    #             'optimizer': optimizer.state_dict(),
    #             # 'lr_scheduler': lr_scheduler,
    #         }, is_best, model_dir)

    # end_time = time.time()
    # print("Total Time:", end_time - start_time)

    if fixed_params.get('print_weights', False):
        if model_rounded is None:
            model_rounded = round_shift_weights(model, clone=True)

        with open(os.path.join(model_dir, 'weights_log.txt'), 'w') as weights_log_file:
            with redirect_stdout(weights_log_file):
                # Log model's state_dict
                print("Model's state_dict:")
                for param_tensor in model_rounded.state_dict():
                    print(param_tensor, "\t", model_rounded.state_dict()[param_tensor].size())
                    print(model_rounded.state_dict()[param_tensor])
                    print("")
    model = model.to(device)

    return model, train_loader, val_loader, criterion, optimizer



def train(train_loader, model, criterion, optimizer, epoch, cfg, fixed_params):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             prefix="Epoch: [{}]".format(epoch))

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if torch.cuda.is_available():
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)


        # Compute output
        output = model(input)
        loss = criterion(output, target)

        # Measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        if cfg['weight_decay'] > 0:
            loss += shift_l2_norm(optimizer, cfg['weight_decay'])
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % fixed_params['print_freq'] == 0:
            progress.print(i)
    
    return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)


def validate(val_loader, model, criterion, cfg, fixed_params):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1,
                             prefix='Test: ')

    # Switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if torch.cuda.is_available():
                input = input.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)


            # Compute output
            output = model(input)
            loss = criterion(output, target)

            # Measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % fixed_params['print_freq'] == 0:
                progress.print(i)

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))

    return (losses.avg, top1.avg.cpu().numpy(), batch_time.avg)

def shift_l2_norm(opt, weight_decay):
    shift_params = opt.param_groups[2]['params']
    l2_norm = 0
    for shift in shift_params:
        l2_norm += torch.sum((2**shift)**2)
    return weight_decay * 0.5 * l2_norm


def save_checkpoint(state, is_best, dir_path, filename='checkpoint.pth.tar'):
    torch.save(state, os.path.join(dir_path, filename))
    if is_best:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    

def create_configspace():
    cs = ConfigurationSpace()
    desired_sums = [8, 16, 32]

    batch_size = Integer("batch_size", (127, 128), default = 128)
    test_batch_size = Integer("test_batch_size", (500, 2000), default = 1000)
    optimizer = Categorical("optimizer", ["SGD", "Adam", "adadelta", "adagrad", "rmsprop", "radam", "ranger"], default = "SGD")
    lr = Float("lr", (0.001, 0.2), default = 0.1)
    momentum = Float("momentum", (0.0, 0.9), default=0.9)
    epochs = Integer("epochs", (2,num_train_epochs), default = 5)
    weight_bits = Integer("weight_bits", (2, 5), default = 5)
    activation_integer_bits = Integer("activation_integer_bits", (4, 32), default = 16)
    activation_fraction_bits = Integer("activation_fraction_bits", (4, 32), default = 16)
    shift_depth = Integer("shift_depth", (1, num_layers), default = num_layers)
    shift_type = Categorical("shift_type", ["Q", "PS"], default = "PS")
    # use_kernel = Categorical("use_kernel", ["False"])
    rounding = Categorical("rounding", ["deterministic", "stochastic"], default = "deterministic")
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default = 1e-4)
    # sum_bits = Integer("sum_bits", lower=min(desired_sums), upper=max(desired_sums))


    cs.add_hyperparameters([batch_size, test_batch_size, optimizer,
                            lr, momentum, epochs, weight_bits,
                            activation_integer_bits, activation_fraction_bits, shift_depth, shift_type, #use_kernel
                            rounding, weight_decay])
    
    # # Define InCondition for sum_bits
    # sum_condition = InCondition(child=sum_bits, parent=activation_integer_bits, values=desired_sums)

    # # Add condition to Configuration Space
    # cs.add_condition(sum_condition)

    return cs


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def plot_pareto(smac: AbstractFacade, incumbents: List[Configuration]) -> None:
    """Plots configurations from SMAC and highlights the best configurations in a Pareto front."""
    average_costs = []
    average_pareto_costs = []
    for config in smac.runhistory.get_configs():
        # Since we use multiple seeds, we have to average them to get only one cost value pair for each configuration
        average_cost = smac.runhistory.average_cost(config)

        if config in incumbents:
            average_pareto_costs += [average_cost]
        else:
            average_costs += [average_cost]

    # Let's work with a numpy array
    costs = np.vstack(average_costs)
    pareto_costs = np.vstack(average_pareto_costs)
    pareto_costs = pareto_costs[pareto_costs[:, 0].argsort()]  # Sort them

    costs_x, costs_y = costs[:, 0], costs[:, 1]
    pareto_costs_x, pareto_costs_y = pareto_costs[:, 0], pareto_costs[:, 1]

    plt.scatter(costs_x, costs_y, marker="x", label="Configuration")
    plt.scatter(pareto_costs_x, pareto_costs_y, marker="x", c="r", label="Incumbent")
    plt.step(
        [pareto_costs_x[0]] + pareto_costs_x.tolist() + [np.max(costs_x)],  # We add bounds
        [np.max(costs_y)] + pareto_costs_y.tolist() + [np.min(pareto_costs_y)],  # We add bounds
        where="post",
        linestyle=":",
    )

    plt.title("Pareto-Front")
    plt.xlabel(smac.scenario.objectives[0])
    plt.ylabel(smac.scenario.objectives[1])
    plt.legend()
   
    # Save the plot to a file
    plt.savefig('/scratch/hpc-prf-intexml/leonahennig/DeepShift/pareto_front.png')

    # Optionally, clear the figure after saving, if you plan to create more plots
    plt.clf()

# train_model function for SMAC optimization
def train_model(config, seed: int = 4, budget: int = 25):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # start_time = time.time()

    # Initialize the emissions tracker
    # tracker = EmissionsTracker()
    tracker = OfflineEmissionsTracker(country_iso_code="CAN")
    # tracker.start()

    # Start tracking
    # tracker.start()
    # Set up model configuration
    model_config = {
        'optimizer': config['optimizer'],
        'lr': config['lr'],
        'momentum': config['momentum'],
        'weight_bits': config['weight_bits'],
        'activation_integer_bits': config['activation_integer_bits'],
        'activation_fraction_bits': config['activation_fraction_bits'],
        'shift_depth': config['shift_depth'],
        'shift_type': config['shift_type'],
        'rounding': config['rounding'],
        'weight_decay': config['weight_decay'],
        'batch_size': config['batch_size'],
        'test_batch_size': config['test_batch_size'],
        'epochs': int(budget),
    }

    # Set up fixed parameters (update as necessary)
    fixed_params = {
        'log_interval': 10,
        # Other fixed parameters...
        # 'gpu': None,  # Assuming no specific GPU
        'dist_url': 'env://',
        'dist_backend': 'nccl',
        'world_size': -1,
        'multiprocessing_distributed': False,
        'evaluate': False,
        'use_kernel': False,
        'save_model': False,
        'pretrained': False,
        'freeze': False,
        'weights': '',
        'workers': 1,
        'resume': '',
        'start_epoch': 0,
        'print_freq': 50
    }

    fixed_params['distributed'] = fixed_params['world_size'] > 1 or fixed_params['multiprocessing_distributed']

    # Set device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    try:
        # with OfflineEmissionsTracker(tracking_mode='process', log_level='critical', country_iso_code="DEU") as tracker:
        # Initialize model using the main_worker function with the new config
        model, train_loader, val_loader, criterion, optimizer = main_worker(0,1,model_config, fixed_params)

        # Train and evaluate the model using existing train and validate functions
        best_acc1 = 0
        for epoch in range(0, model_config['epochs']):
            train(train_loader, model, criterion, optimizer, epoch, model_config, fixed_params)
            with EmissionsTracker(tracking_mode='process', log_level='critical') as tracker:
                val_log = validate(val_loader, model, criterion, model_config, fixed_params)
                acc1 = val_log[1]  # Assuming acc1 is the second value in val_log

                # Update best accuracy
        best_acc1 = max(acc1, best_acc1)

        # emissions = tracker.stop()
                    
                
        test_energy = float(tracker._total_energy)

        return {
                "loss": 1 - np.divide(best_acc1,100),  # Assuming best_acc1 is your accuracy metric
                "emissions": test_energy,  # Emissions measured
            }
    except Exception as e:
        print(e)
        return {
                "loss": float(1),
                "emissions": float(1)
            }
    except torch.cuda.CudaError as e:
        print(f"CUDA error encountered: {e}")
        return {
                "loss": float(1),
                "emissions": float(1)
            }  # Return a high loss value if a critical CUDA error occurs
    except Exception as e:
        print(f"Error encountered: {e}")
        return {
                "loss": float(1),
                "emissions": float(1)
            }  # Return a high loss value if another error occurs




def main():
    cs = create_configspace()
    # objectives = ["accuracy", "time"]
    objectives = ["loss", "emissions"]
    facades: list[AbstractFacade] = []

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    timestamp = current_datetime.strftime("%Y%m%d%H%M%S")

    scenario = Scenario(
        cs,
        objectives = objectives,
        # trial_walltime_limit=3500,  # Set a suitable time limit for each trial
        n_trials=100,  # Total number of configurations to try
        min_budget=2,  # Minimum number of epochs for training
        max_budget=num_train_epochs,  # Maximum number of epochs for training
        n_workers=1,  # Number of parallel workers (set based on available resources)
        use_default_config = True,
        name=f'MO-codecarbon{timestamp}_caltech101_mobilenet',
        seed=12
        # crash_cost=[float(1000),float(1000)],
        # termination_cost_threshold=[float(1001),float(1001)]
    )

    # Create the intensifier object
    intensifier_object = Hyperband
    intensifier = intensifier_object(scenario, incumbent_selection="any_budget")

    additional_config = Configuration(cs, {
        "batch_size": 128,
        "test_batch_size": 1000,
        "optimizer": "Adam",
        "lr": 0.1,
        "momentum": 0.9,
        "epochs": 10,
        "weight_bits": 4,
        "activation_integer_bits": 16,
        "activation_fraction_bits": 16,
        "shift_depth": num_layers,
        "shift_type": "PS",
        "rounding": "stochastic",
        "weight_decay": 1e-4
    })

    additional_configs = [additional_config]

    # Initial design for random configurations
    initial_design = MFFacade.get_initial_design(scenario, n_configs=15, additional_configs=additional_configs)

    multi_objective_algorithm = ParEGO(scenario)

    # Run the optimization with different seeds
    smac = MFFacade(
        scenario,
        train_model,  # Pass the seed here
        initial_design=initial_design,
        multi_objective_algorithm=multi_objective_algorithm,
        intensifier=intensifier,
        overwrite=True
    )

    # Optimize to find the best configuration
    incumbents = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Validated costs from default config: \n--- {default_cost}\n")

    # Let's calculate the cost of the incumbent
    print("Validated costs from the Pareto front (incumbents):")
    for incumbent in incumbents:
        cost = smac.validate(incumbent)
        print("---", cost)

    print(incumbents)

if __name__ == "__main__":
    main()


