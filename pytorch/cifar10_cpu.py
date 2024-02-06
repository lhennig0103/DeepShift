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

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float


import torchsummary
import optim

from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from unoptimized.convert import convert_to_unoptimized
import unoptimized

import matplotlib.pyplot as plt
from typing import List
import multiprocessing as mp
import numpy as np

from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float, InCondition

from functools import partial

import cifar10_models as models

'''
Unfortunately, none of the pytorch repositories with ResNets on CIFAR10 provides an 
implementation as described in the original paper. If you just use the torchvision's 
models on CIFAR10 you'll get the model that differs in number of layers and parameters. 
This is unacceptable if you want to directly compare ResNet-s on CIFAR10 with the 
original paper. The purpose of resnet_cifar10 (which has been obtained from https://github.com/akamaster/pytorch_resnet_cifar10
is to provide a valid pytorch implementation of ResNet-s for CIFAR10 as described in the original paper. 
'''
device = torch.device("cpu")

# Set the start method for multiprocessing
if __name__ == '__main__':
    mp.set_start_method('spawn')
    
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')

best_acc1 = 0
args = parser.parse_args()

# Global variable for logging
optimization_log = []


def main_worker(cfg, fixed_params):
    global best_acc1

    # Create model
    print(f"=> creating model '{args.arch}'")
    model = models.__dict__[args.arch]()
    model = model.to(device)

    model_rounded = None

    if 'weights' in fixed_params and fixed_params['weights']:
        saved_weights = torch.load(fixed_params['weights'], map_location=device)
        if isinstance(saved_weights, nn.Module):
            state_dict = saved_weights.state_dict()
        elif "state_dict" in saved_weights:
            state_dict = saved_weights["state_dict"]
        else:
            state_dict = saved_weights

        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' of dataparallel
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    if cfg['shift_depth'] > 0:
        model, _ = convert_to_shift(model, cfg['shift_depth'], cfg['shift_type'],
                                    convert_weights=(fixed_params['pretrained'] or fixed_params['weights']),
                                    rounding=cfg.get('rounding', 'deterministic'),
                                    weight_bits=cfg.get('weight_bits', 5),
                                    act_integer_bits=cfg.get('activation_integer_bits', 16),
                                    act_fraction_bits=cfg.get('activation_fraction_bits', 16))
    elif cfg.get('shift_depth', 0) == 0 and not fixed_params['use_kernel']:
        model = convert_to_unoptimized(model)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = torch.optim.SGD(model.parameters(), cfg['lr'],
                                momentum=cfg['momentum'],
                                weight_decay=cfg['weight_decay'])

    # Data loading code
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True,
                                     transform=transforms.Compose([
                                         transforms.RandomHorizontalFlip(),
                                         transforms.RandomCrop(32, 4),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                     ]))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10(root="./data", train=False, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])),
        batch_size=cfg['batch_size'], shuffle=False)

    for epoch in range(cfg['epochs']):
        train(train_loader, model, criterion, optimizer, epoch, cfg)
        acc1 = validate(val_loader, model, criterion)

        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        if is_best:
            print(f"New best accuracy: {best_acc1}")
            # Save best model weights here if needed

    return best_acc1


def train(train_loader, model, criterion, optimizer, epoch, cfg):
    model.train()
    for i, (input, target) in enumerate(train_loader):
        input, target = input.to(device), target.to(device)

        # Compute output
        output = model(input)
        loss = criterion(output, target)

        # Compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t Loss {loss.item():.4f}")

def validate(val_loader, model, criterion):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)
            output = model(input)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    acc1 = 100 * correct / total
    print(f'Accuracy of the network on the 10000 test images: {acc1} %')
    return acc1

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
    batch_size = Integer("batch_size", (32, 512), default = 128)
    test_batch_size = Integer("test_batch_size", (500, 2000), default = 1000)
    optimizer = Categorical("optimizer", ["SGD", "Adam", "adadelte", "adagrad", "rmsprop", "radam", "ranger"], default = "SGD")
    lr = Float("lr", (0.001, 0.2), default = 0.1)
    momentum = Float("momentum", (0.0, 0.9), default=0.9)
    epochs = Integer("epochs", (40, 150), default = 80)
    weight_bits = Integer("weight_bits", (2, 8), default = 5)
    activation_integer_bits = Integer("activation_integer_bits", (2, 32), default = 16)
    activation_fraction_bits = Integer("activation_fraction_bits", (2, 32), default = 16)
    shift_depth = Integer("shift_depth", (1, 20), default = 20)
    shift_type = Categorical("shift_type", ["Q", "PS"], default = "PS")
    # use_kernel = Categorical("use_kernel", ["False"])
    rounding = Categorical("rounding", ["deterministic", "stochastic"], default = "deterministic")
    weight_decay = Float("weight_decay", (1e-6, 1e-2), default = 1e-4)
    # sum_bits = Integer("sum_bits", (desired_sums, desired_sums))



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
    
def plot_trajectory():
    times = [entry['time'] for entry in optimization_log]
    values = [entry['performance'] for entry in optimization_log]

    plt.figure()
    plt.title("Optimization Trajectory")
    plt.xlabel("Wallclock Time (s)")
    plt.ylabel("Objective Value")
    plt.plot(times, values, label='Trajectory')
    plt.legend()
    plt.savefig("/scratch/hpc-prf-intexml/leonahennig/DeepShift/cifar10.png")

# [Imports and definitions as in your previous script]

def train_model(config, seed: int = 4, budget: int = 25):
    try:
        # Set up model configuration
        np.random.seed(seed=seed)
            
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
            'epochs': int(budget)
        }

        # Set up fixed parameters (update as necessary)
        fixed_params = {
            'log_interval': 10,
            # Other fixed parameters...
            'dist_url': 'env://',
            'dist_backend': 'nccl',
            'world_size': -1,
            'multiprocessing_distributed': False,
            'evaluate': False,
            'use_kernel': False,
            'save_model': False,
            'pretrained': True,
            'freeze': False,
            'weights': '',
            'workers': 4,
            'resume': '',
            'start_epoch': 0,
            'print_freq': 50
        }
        fixed_params['distributed'] = fixed_params['world_size'] > 1 or fixed_params['multiprocessing_distributed']

        # Set device for training
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(seed)

        # Initialize model using the main_worker function with the new config
        model, train_loader, val_loader, criterion, optimizer = main_worker(model_config, fixed_params)

        # Train and evaluate the model
        best_acc1 = 0
        for epoch in range(fixed_params['start_epoch'], model_config['epochs']):
            train(train_loader, model, criterion, optimizer, epoch, model_config, fixed_params)
            val_log = validate(val_loader, model, criterion, model_config, fixed_params)
            acc1 = val_log[1]  # Assuming acc1 is the second value in val_log

            # Update best accuracy
            best_acc1 = max(acc1, best_acc1)

        # Check if the accuracy is a finite number
        if not np.isfinite(acc1):
            raise ValueError("Non-finite accuracy detected.")

        # Return best accuracy as the metric for SMAC
        return 1 - best_acc1

    except Exception as e:
        print(f"An error occurred during model training or evaluation: {e}")
        # Optionally, log the error details to a file or external logging system here


def main():
    global best_acc1

    # Define fixed parameters (you might need to modify this based on your script's needs)
    fixed_params = {
        # 'seed': random.randint(0, 2**32 - 1),
        'gpu': None,  # Assuming you want to keep it flexible; set to a specific GPU ID if needed
        'dist_url': 'tcp://224.66.41.62:23456',
        'dist_backend': 'nccl',
        'world_size': -1,
        'multiprocessing_distributed': False,
        'evaluate': False,
        'use_kernel': False,
        'save_model': False,
        'pretrained': True,
        'freeze': False,
        'weights': '',
        'workers': 4,
        'resume': '',
        'start_epoch': 0,
        'print_freq': 50
    }

    # Place the code snippet here
    if not fixed_params['evaluate'] and fixed_params['use_kernel']:
        raise ValueError('Our custom kernel currently supports inference only, not training.')

    # if fixed_params['seed'] is not None:
    #     random.seed(fixed_params['seed'])
    #     torch.manual_seed(fixed_params['seed'])
    #     cudnn.deterministic = True
    #     warnings.warn('You have chosen to seed training. This will turn on the CUDNN deterministic setting, which can slow down your training considerably!')

    if fixed_params['gpu'] is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if fixed_params['dist_url'] == "env://" and fixed_params['world_size'] == -1:
        fixed_params['world_size'] = int(os.environ["WORLD_SIZE"])

    fixed_params['distributed'] = fixed_params['world_size'] > 1 or fixed_params['multiprocessing_distributed']

    # if fixed_params['multiprocessing_distributed']:
    #     # Adjust world size and launch distributed processes
    #     fixed_params['world_size'] = ngpus_per_node * fixed_params['world_size']
    #     mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, cfg, fixed_params))
    # else:
    #     # Call main_worker function directly
    #     main_worker(fixed_params['gpu'], ngpus_per_node, cfg, fixed_params)

    cs = create_configspace()

    facades: list[AbstractFacade] = []

    # train_budgeted_model = partial(train_model, budget=10, seed=1234)

    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Format the date and time as a string
    timestamp = current_datetime.strftime("%Y%m%d%H%M%S")

    scenario = Scenario(
        cs,
        trial_walltime_limit=2000,  # Set a suitable time limit for each trial
        n_trials=200,  # Total number of configurations to try
        min_budget=3,  # Minimum number of epochs for training
        max_budget=15,  # Maximum number of epochs for training
        n_workers=1,  # Number of parallel workers (set based on available resources)
        use_default_config = True,
        name=f'cifar10_mf_epochs{timestamp}'
    )

    # Create the intensifier object
    intensifier_object = Hyperband
    intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

    # Initial design for random configurations
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    smac = MFFacade(
        scenario,
        train_model,
        initial_design=initial_design,
        intensifier=intensifier,
        overwrite=True,
        )   

    # Optimize to find the best configuration
    incumbent = smac.optimize()

    # Evaluate the default configuration
    default_cost = smac.validate(cs.get_default_configuration())

    # Evaluate the best found configuration
    incumbent_cost = smac.validate(incumbent)
    
    # Print results
    print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")
    print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")
    print(incumbent, smac.validate)

    facades.append(smac)

    # Print the best configuration and plot the optimization trajectory
    plot_trajectory(facades)

if __name__ == "__main__":
    main()