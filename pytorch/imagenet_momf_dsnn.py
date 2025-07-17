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
import tinyimagenet_models as models
from torch.utils.data import DataLoader

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Configuration, Categorical, Integer, Float, InCondition
from ConfigSpace.conditions import AbstractCondition
from smac.multi_objective.parego import ParEGO
import numpy as np
from codecarbon import EmissionsTracker
from codecarbon import OfflineEmissionsTracker
import datetime

from pathlib import Path
from tinyimagenet import TinyImageNet

import torchsummary
import optim
import copy

from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from unoptimized.convert import convert_to_unoptimized
import unoptimized



import customized_models

# export TORCH_USE_CUDA_DSA=1


default_model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

customized_models_names = sorted(name for name in customized_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(customized_models.__dict__[name]))

for name in customized_models.__dict__:
    if name.islower() and not name.startswith("__") and callable(customized_models.__dict__[name]):
        models.__dict__[name] = customized_models.__dict__[name]

model_names = default_model_names + customized_models_names

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

num_train_epochs = 15
num_layers=19
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# parser.add_argument('data', metavar='DIR',
#                     help='path to dataset')
# parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
#                     choices=model_names,
#                     help='model architecture: ' +
#                         ' | '.join(model_names) +
#                         ' (default: resnet18)')
# parser.add_argument('--model', default='', type=str, metavar='MODEL_PATH',
#                     help='path to model file to load both its architecture and weights (default: none)')
# parser.add_argument('--weights', default='', type=str, metavar='WEIGHTS_PATH',
#                     help='path to file to load its weights (default: none)')
# parser.add_argument('-s', '--shift-depth', type=int, default=0,
#                     help='how many layers to convert to shift')
# parser.add_argument('-st', '--shift-type', default='PS', choices=['Q', 'PS'],
#                     help='type of DeepShift method for training and representing weights (default: PS)')
# parser.add_argument('-r', '--rounding', default='deterministic', choices=['deterministic', 'stochastic'],
#                     help='type of rounding (default: deterministic)')
# parser.add_argument('-wb', '--weight-bits', type=int, default=5,
#                     help='number of bits to represent the weights')
# parser.add_argument('-ab', '--activation-bits', nargs='+', default=[16,16],
#                     help='number of integer and fraction bits to represent activation (fixed point format)') 
# parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
#                     help='number of data loading workers (default: 4)')
# parser.add_argument('--epochs', default=90, type=int, metavar='N',
#                     help='number of total epochs to run')
# parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
#                     help='manual epoch number (useful on restarts)')
# parser.add_argument('-opt', '--optimizer', metavar='OPT', default="SGD", 
#                     help='optimizer algorithm')
# parser.add_argument('-b', '--batch-size', default=256, type=int,
#                     metavar='N',
#                     help='mini-batch size (default: 256), this is the total '
#                          'batch size of all GPUs on the current node when '
#                          'using Data Parallel or Distributed Data Parallel')
# parser.add_argument('-bm', '--batch-multiplier', default=1, type=int,
#                     help='how many batches to repeat before updating parameter. '
#                          'effective batch size is batch-size * batch-multuplier')
# parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
#                     metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--lr-schedule', dest='lr_schedule', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
#                     help='using learning rate schedule')
# parser.add_argument('--lr-step-size', default=30, type=int,
#                     help='epoch numbers at which to decay learning rate (only applicable if --lr-schedule is set to StepLR)', dest='lr_step_size')
# parser.add_argument('--lr-sign', default=None, type=float,
#                     help='separate initial learning rate for sign params')
# parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
#                     help='momentum')
# parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
#                     metavar='W', help='weight decay (default: 1e-4)',
#                     dest='weight_decay')
# parser.add_argument('--opt-ckpt', default='', type=str, metavar='OPT_PATH',
#                     help='path to checkpoint file to load optimizer state (default: none)')
# parser.add_argument('-p', '--print-freq', default=10, type=int,
#                     metavar='N', help='print frequency (default: 10)')
# parser.add_argument('--resume', default='', type=str, metavar='CHECKPOINT_PATH',
#                     help='path to latest checkpoint (default: none)')
# parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
#                     help='only evaluate model on validation set')
# parser.add_argument('--pretrained', dest='pretrained', default=False, type=lambda x:bool(distutils.util.strtobool(x)), 
#                     help='use pre-trained model')
# parser.add_argument('--world-size', default=-1, type=int,
#                     help='number of nodes for distributed training')
# parser.add_argument('--rank', default=-1, type=int,
#                     help='node rank for distributed training')
# parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
#                     help='url used to set up distributed training')
# parser.add_argument('--dist-backend', default='nccl', type=str,
#                     help='distributed backend')
# parser.add_argument('--seed', default=None, type=int,
#                     help='seed for initializing training. ')
# parser.add_argument('--gpu', default=None, type=int,
#                     help='GPU id to use.')
# parser.add_argument('--multiprocessing-distributed', action='store_true',
#                     help='Use multi-processing distributed training to launch '
#                          'N processes per node, which has N GPUs. This is the '
#                          'fastest way to use PyTorch for either single node or '
#                          'multi node data parallel training')

# parser.add_argument('--save-model', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
#                     help='For Saving the current Model (default: True)')
# parser.add_argument('--print-weights', default=True, type=lambda x:bool(distutils.util.strtobool(x)), 
#                     help='For printing the weights of Model (default: True)')
# parser.add_argument('--desc', type=str, default=None,
#                     help='description to append to model directory name')
# parser.add_argument('--use-kernel', type=lambda x:bool(distutils.util.strtobool(x)), default=False,
#                     help='whether using custom shift kernel')

best_acc1 = 0


# def main():
#     args = parser.parse_args()

#     if(args.evaluate is False and args.use_kernel is True):
#         raise ValueError('Our custom kernel currently supports inference only, not training.')

#     if args.seed is not None:
#         random.seed(args.seed)
#         torch.manual_seed(args.seed)
#         cudnn.deterministic = True
#         warnings.warn('You have chosen to seed training. '
#                       'This will turn on the CUDNN deterministic setting, '
#                       'which can slow down your training considerably! '
#                       'You may see unexpected behavior when restarting '
#                       'from checkpoints.')

#     if args.gpu is not None:
#         warnings.warn('You have chosen a specific GPU. This will completely '
#                       'disable data parallelism.')

#     if args.dist_url == "env://" and args.world_size == -1:
#         args.world_size = int(os.environ["WORLD_SIZE"])

#     args.distributed = args.world_size > 1 or args.multiprocessing_distributed

#     assert len(cfg['activation_bits'])==2, "activation-bits argument needs to be a tuple of 2 values representing number of integer bits and number of fraction bits, e.g., '3 5' for 8-bits fixed point or '3 13' for 16-bits fixed point"
#     [args.activation_integer_bits, args.activation_fraction_bits] = args.activation_bits
#     [args.activation_integer_bits, args.activation_fraction_bits] = [int(args.activation_integer_bits), int(args.activation_fraction_bits)]

#     ngpus_per_node = torch.cuda.device_count()
#     if args.multiprocessing_distributed:
#         # Since we have ngpus_per_node processes per node, the total world_size
#         # needs to be adjusted accordingly
#         args.world_size = ngpus_per_node * args.world_size
#         # Use torch.multiprocessing.spawn to launch distributed processes: the
#         # main_worker process function
#         mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
#     else:
#         # Simply call main_worker function
#         main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, cfg, fixed_params):
    global best_acc1
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
                                    act_integer_bits=cfg.get('activation_integer_bits', 16), 
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

    # lr_scheduler = None
    # if args.opt_ckpt:
    #     print("WARNING: Ignoring arguments \"lr\", \"momentum\", \"weight_decay\", and \"lr_schedule\"")

    #     opt_ckpt = torch.load(args.opt_ckpt)
    #     if 'optimizer' in opt_ckpt:
    #         opt_ckpt = opt_ckpt['optimizer']
    #     optimizer.load_state_dict(opt_ckpt)

    #     if 'lr_scheduler' in opt_ckpt:
    #         lr_scheduler = opt_ckpt['lr_scheduler']

    # TODO: enable different lr scheduling algorithms 
    #if (args.lr_schedule and lr_scheduler is not None):
    #    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
    #                                                        milestones=[100, 150], last_epoch=args.start_epoch - 1)

    # # optionally resume from a checkpoint
    # if args.resume:
    #     if os.path.isfile(args.resume):
    #         print("=> loading checkpoint '{}'".format(args.resume))
    #         checkpoint = torch.load(args.resume)
    #         args.start_epoch = checkpoint['epoch']
    #         best_acc1 = checkpoint['best_acc1']
    #         if args.gpu is not None:
    #             # best_acc1 may be from a checkpoint from a different GPU
    #             best_acc1 = best_acc1.to(args.gpu)
    #         try:
    #             model.load_state_dict(checkpoint['state_dict'])
    #         except:
    #             # create new OrderedDict that does not contain module.
    #             state_dict = checkpoint['state_dict']
    #             new_state_dict = OrderedDict()
    #             for k, v in state_dict.items():
    #                 if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
    #                     if (k.startswith("features")):
    #                         name = k[0:9] + k[9+7:] # remove "module" after features
    #                     else:
    #                         name = k
    #                 else:
    #                     name = k[7:] # remove "module" at beginning of name
    #                 new_state_dict[name] = v
                
    #             # load params
    #             model.load_state_dict(new_state_dict)
    #         optimizer.load_state_dict(checkpoint['optimizer'])
    #         if 'lr_scheduler' in checkpoint and checkpoint['lr_scheduler'] is not None:
    #             lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #         print("=> loaded checkpoint '{}' (epoch {})"
    #               .format(args.resume, checkpoint['epoch']))
    #     else:
    #         print("=> no checkpoint found at '{}'".format(args.resume))

    # If evaluating, round weights to ensure that the results are due to powers of 2 weights
    if fixed_params.get('evaluate', False):
        model = round_shift_weights(model)

    cudnn.benchmark = True

    model_summary = None
    try:
        model_summary, model_params_info = torchsummary.summary_string(model, input_size=(3,228,228))
        print(model_summary)
        print("WARNING: The summary function reports duplicate parameters for multi-GPU case")
    except:
        print("WARNING: Unable to obtain summary of model")

    cudnn.benchmark = True

    # Naming model sub-directory
    conv2d_layers_count = count_layer_type(model, nn.Conv2d) + count_layer_type(model, unoptimized.UnoptimizedConv2d)
    linear_layers_count = count_layer_type(model, nn.Linear) + count_layer_type(model, unoptimized.UnoptimizedLinear)
    
    shift_label = "shift" if cfg['shift_depth'] == 0 else "shift_ps" if cfg['shift_type'] == 'PS' else "shift_q"
    shift_label += "_all" if conv2d_layers_count == 0 and linear_layers_count == 0 else "_%s" % cfg['shift_depth']
    shift_label += "_wb_%s" % cfg['weight_bits'] if cfg['shift_depth'] > 0 else ""

    # 'desc_label = "_%s" % cfg['desc'] if cfg.get('desc') else ""'

    model_name = '%s/%s' % (args.arch, shift_label)

    if fixed_params['save_model'] == False:
        model_dir = os.path.join(os.path.join(os.path.join(os.getcwd(), "models"), "tinyimagenet"), model_name)
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
    
    data_dir = "/pc2/users/i/inxml10/.torchvision/tinyimagenet"

    transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize to TinyImageNet's image size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
    split = "train"  # Specify which split to load (train or val)
    train_dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), transform=transform, split=split)

    split = "val"  # Specify which split to load (train or val)
    val_dataset = TinyImageNet(Path("~/.torchvision/tinyimagenet/"), transform=transform, split=split)



    # traindir = os.path.join(data_dir, 'train')
    # valdir = os.path.join(data_dir, 'val')
    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                  std=[0.229, 0.224, 0.225])

    # train_dataset = datasets.ImageFolder(traindir,
    #     transforms.Compose([
    #         transforms.RandomResizedCrop(224),
    #         transforms.RandomHorizontalFlip(),
    #         transforms.ToTensor(),
    #         normalize,
    #     ]))

    train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg['batch_size'], shuffle=(train_sampler is None),
        num_workers=0, pin_memory=True, sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=cfg['batch_size'], shuffle=False,
        num_workers=0, pin_memory=True)


    start_time = time.time()

    # if args.evaluate:
    #     start_log_time = time.time()
    #     val_log = validate(val_loader, model, criterion, args)
    #     val_log = [val_log]

    #     with open(os.path.join(model_dir, "test_log.csv"), "w") as test_log_file:
    #         test_log_csv = csv.writer(test_log_file)
    #         test_log_csv.writerow(['test_loss', 'test_top1_acc', 'test_top5_acc', 'test_time', 'cumulative_time'])
    #         test_log_csv.writerows(val_log + [(time.time() - start_log_time,)])
    # else:
    #     train_log = []

    #     with open(os.path.join(model_dir, "train_log.csv"), "w") as train_log_file:
    #         train_log_csv = csv.writer(train_log_file)
    #         train_log_csv.writerow(['epoch', 'train_loss', 'train_top1_acc', 'train_top5_acc', 'train_time', 'test_loss', 'test_top1_acc', 'test_top5_acc', 'test_time', 'cumulative_time'])

    # for epoch in range(fixed_params['start_epoch'], cfg['epochs']):
    #     if fixed_params['distributed']:
    #         train_sampler.set_epoch(epoch)

        # if cfg.get('alternate_update'):
        #     if epoch % 2 == 1:
        #         optimizer.param_groups[1]['lr'] = 0
        #         optimizer.param_groups[2]['lr'] = optimizer.param_groups[0]['lr']
        #     else:
        #         optimizer.param_groups[1]['lr'] = optimizer.param_groups[0]['lr']
        #         optimizer.param_groups[2]['lr'] = 0

        # Train for one epoch
        # print("current lr ", [param['lr'] for param in optimizer.param_groups])
        # train_epoch_log = train(train_loader, model, criterion, optimizer, epoch, cfg, fixed_params)
        # if cfg.get('lr_schedule'):
        #     lr_scheduler.step()

        # Evaluate on validation set
        # val_epoch_log = validate(val_loader, model, criterion, cfg, fixed_params)
        # acc1 = val_epoch_log[2]

        # # Append to log
        # with open(os.path.join(model_dir, "train_log.csv"), "a") as train_log_file:
        #     train_log_csv = csv.writer(train_log_file)
        #     train_log_csv.writerow(((epoch,) + train_epoch_log + val_epoch_log)) 

        # # Remember best acc@1 and save checkpoint
        # is_best = acc1 > best_acc1
        # best_acc1 = max(acc1, best_acc1)

        # if fixed_params.get('print_weights'):
        #     with open(os.path.join(model_dir, 'weights_log_' + str(epoch) + '.txt'), 'w') as weights_log_file:
        #         with redirect_stdout(weights_log_file):
        #             print("Model's state_dict:")
        #             for param_tensor in model.state_dict():
        #                 print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        #                 print(model.state_dict()[param_tensor])

        # if not fixed_params['multiprocessing_distributed'] or (fixed_params['multiprocessing_distributed'] and fixed_params['rank'] % ngpus_per_node == 0):
        #     if is_best:
        #         try:
        #             if cfg.get('save_model'):
        #                 model_rounded = round_shift_weights(model, clone=True)
        #                 torch.save(model_rounded.state_dict(), os.path.join(model_dir, "weights.pth"))
        #                 torch.save(model_rounded, os.path.join(model_dir, "model.pth"))
        #         except: 
        #             print("WARNING: Unable to save model.pth")

        #     save_checkpoint({
        #         'epoch': epoch + 1,
        #         'arch': args.arch,
        #         'state_dict': model.state_dict(),
        #         'best_acc1': best_acc1,
        #         'optimizer': optimizer.state_dict(),
        #         # 'lr_scheduler': lr_scheduler,
        #     }, is_best, model_dir)

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

    if (state['epoch']-1)%10 == 0:
        shutil.copyfile(os.path.join(dir_path, filename), os.path.join(dir_path, 'checkpoint_' + str(state['epoch']-1) + '.pth.tar'))    


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


def adjust_learning_rate(optimizer, epoch, initial_lr, step_size=30):
    """Sets the learning rate to the initial LR decayed by 10 every step_size epochs"""
    lr = initial_lr * (0.1 ** (epoch // step_size))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


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
    activation_integer_bits = Integer("activation_integer_bits", (2, 32), default = 16)
    activation_fraction_bits = Integer("activation_fraction_bits", (2, 32), default = 16)
    shift_depth = Integer("shift_depth", (0, num_layers), default = num_layers)
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


# train_model function for SMAC optimization
def train_model(config, seed: int = 4, budget: int = 25):
    # Set seeds for reproducibility
    # torch.manual_seed(seed)
    # np.random.seed(seed)
    # random.seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
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
    # torch.manual_seed(seed)

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
        n_trials=80,  # Total number of configurations to try
        min_budget=4,  # Minimum number of epochs for training
        max_budget=num_train_epochs,  # Maximum number of epochs for training
        n_workers=1,  # Number of parallel workers (set based on available resources)
        use_default_config = True,
        name=f'MO-codecarbon{timestamp}_imagenet_resnet20',
        seed=3
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
    initial_design = MFFacade.get_initial_design(scenario, n_configs=25, additional_configs=additional_configs)

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



