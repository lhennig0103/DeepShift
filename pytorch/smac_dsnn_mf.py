# DSNN imports
from __future__ import print_function
import warnings
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import optim
from torchvision import datasets, transforms
import csv
import distutils
import distutils.util
import os
from contextlib import redirect_stdout
import time
import torchsummary
import mnist
import copy
from typing import List
from matplotlib import pyplot as plt

import deepshift
import unoptimized
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from unoptimized.convert import convert_to_unoptimized

import numpy as np
from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)

print(torch.cuda.is_available())


class LinearMNIST(nn.Module):
    def __init__(self):
        super(LinearMNIST, self).__init__()
        self.fc1 = nn.Linear(1 * 28 * 28, 512)
        self.dropout1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(512, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class ConvMNIST(nn.Module):
    def __init__(self):
        super(ConvMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class DSNN:
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--type",
        default="linear",
        choices=["linear", "conv"],
        help="model architecture type: "
        + " | ".join(["linear", "conv"])
        + " (default: linear)",
    )
    parser.add_argument(
        "--model",
        default="",
        type=str,
        metavar="MODEL_PATH",
        help="path to model file to load both its architecture and weights (default: none)",
    )
    parser.add_argument(
        "--weights",
        default="",
        type=str,
        metavar="WEIGHTS_PATH",
        help="path to file to load its weights (default: none)",
    )
    parser.add_argument(
        "--shift-depth", type=int, default=0, help="how many layers to convert to shift"
    )
    parser.add_argument(
        "-st",
        "--shift-type",
        default="PS",
        choices=["Q", "PS"],
        help="type of DeepShift method for training and representing weights (default: PS)",
    )
    parser.add_argument(
        "-r",
        "--rounding",
        default="deterministic",
        choices=["deterministic", "stochastic"],
        help="type of rounding (default: deterministic)",
    )
    parser.add_argument(
        "-wb",
        "--weight-bits",
        type=int,
        default=5,
        help="number of bits to represent the shift weights",
    )
    parser.add_argument(
        "-ab",
        "--activation-bits",
        nargs="+",
        default=[16, 16],
        help="number of integer and fraction bits to represent activation (fixed point format)",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=1,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 1)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "-opt", "--optimizer", metavar="OPT", default="SGD", help="optimizer algorithm"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.0,
        metavar="M",
        help="SGD momentum (default: 0.0)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="CHECKPOINT_PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="only evaluate model on validation set",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        default=False,
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="use pre-trained model of full conv or fc model",
    )

    parser.add_argument(
        "--save-model",
        default=True,
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="For Saving the current Model (default: True)",
    )
    parser.add_argument(
        "--print-weights",
        default=True,
        type=lambda x: bool(distutils.util.strtobool(x)),
        help="For printing the weights of Model (default: True)",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default=None,
        help="description to append to model directory name",
    )
    parser.add_argument(
        "--use-kernel",
        type=lambda x: bool(distutils.util.strtobool(x)),
        default=False,
        help="whether using custom shift kernel",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

    if args.model:
        if args.type or args.pretrained:
            print(
                'WARNING: Ignoring arguments "type" and "pretrained" when creating model...'
            )
        model = None
        saved_checkpoint = torch.load(args.model)
        if isinstance(saved_checkpoint, nn.Module):
            model = saved_checkpoint
        elif "model" in saved_checkpoint:
            model = saved_checkpoint["model"]
        else:
            raise Exception("Unable to load model from " + args.model)
    else:
        if args.type == "linear":
            model = LinearMNIST().to(device)
        elif args.type == "conv":
            model = ConvMNIST().to(device)

        if args.pretrained:
            model.load_state_dict(
                torch.load(
                    "./models/mnist/simple_" + args.type + "/shift_0/weights.pth"
                )
            )
            model = model.to(device)

    model_rounded = None

    if args.weights:
        saved_weights = torch.load(args.weights)
        if isinstance(saved_weights, nn.Module):
            state_dict = saved_weights.state_dict()
        elif "state_dict" in saved_weights:
            state_dict = saved_weights["state_dict"]
        else:
            state_dict = saved_weights

        model.load_state_dict(state_dict)

    if args.shift_depth > 0:
        model, _ = convert_to_shift(
            model,
            args.shift_depth,
            args.shift_type,
            convert_all_linear=(args.type != "linear"),
            convert_weights=True,
            use_kernel=args.use_kernel,
            use_cuda=use_cuda,
            rounding=args.rounding,
            weight_bits=args.weight_bits,
            act_integer_bits=args.activation_integer_bits,
            act_fraction_bits=args.activation_fraction_bits,
        )
        model = model.to(device)
    elif args.use_kernel and args.shift_depth == 0:
        model = convert_to_unoptimized(model)
        model = model.to(device)
    elif args.use_kernel and args.shift_depth == 0:
        model = convert_to_unoptimized(model)
        model = model.to(device)

    loss_fn = F.cross_entropy  # F.nll_loss
    # define optimizer
    optimizer = None
    if args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum)
    elif args.optimizer.lower() == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), args.lr)
    elif args.optimizer.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
    elif args.optimizer.lower() == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
    elif args.optimizer.lower() == "radam":
        optimizer = optim.RAdam(model.parameters(), args.lr)
    elif args.optimizer.lower() == "ranger":
        optimizer = optim.Ranger(model.parameters(), args.lr)
    else:
        raise ValueError(
            "Optimizer type: ", args.optimizer, " is not supported or known"
        )

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            if "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            print("=> loaded checkpoint '{}'".format(args.resume))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # name model sub-directory "shift_all" if all layers are converted to shift layers
    conv2d_layers_count = count_layer_type(model, nn.Conv2d) + count_layer_type(
        model, unoptimized.UnoptimizedConv2d
    )
    linear_layers_count = count_layer_type(model, nn.Linear) + count_layer_type(
        model, unoptimized.UnoptimizedLinear
    )
    if args.shift_depth > 0:
        if args.shift_type == "Q":
            shift_label = "shift_q"
        else:
            shift_label = "shift_ps"
    else:
        shift_label = "shift"

    # name model sub-directory "shift_all" if all layers are converted to shift layers
    conv2d_layers_count = count_layer_type(model, nn.Conv2d)
    linear_layers_count = count_layer_type(model, nn.Linear)
    if conv2d_layers_count == 0 and linear_layers_count == 0:
        shift_label += "_all"
    else:
        shift_label += "_%s" % (args.shift_depth)

    if args.shift_depth > 0:
        shift_label += "_wb_%s" % (args.weight_bits)

    if args.desc is not None and len(args.desc) > 0:
        desc_label = "_%s" % (args.desc)
    else:
        desc_label = ""

    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges.
        # To illustrate different parameter types, we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace()

        model_type = Categorical("model_type", ["linear", "conv"], default="linear")
        batch_size = Integer("batch_size", (32, 128), default=64)
        test_batch_size = Integer("test_batch_size", (500, 2000), default=1000)
        optimizer_type = Categorical("optimizer_type", ["SGD", "Adam"], default="SGD")
        learning_rate = Float("learning_rate", (0.001, 0.1), default=0.01)
        momentum = Float("momentum", (0.0, 0.9), default=0.5)
        num_epochs = Integer("num_epochs", (5, 20), default=10)
        weight_bits = Integer("weight_bits", (2, 8), default=5)
        activation_integer_bits = Integer(
            "activation_integer_bits", (2, 32), default=16
        )
        activation_fraction_bits = Integer(
            "activation_fraction_bits", (2, 32), default=16
        )
        shift_depth = Integer("shift_depth", (0, 1500), default=100)

        # Add all hyperparameters at once:
        cs.add_hyperparameters(
            [
                model_type,
                batch_size,
                test_batch_size,
                optimizer_type,
                learning_rate,
                momentum,
                num_epochs,
                weight_bits,
                activation_integer_bits,
                activation_fraction_bits,
                shift_depth,
            ]
        )

        # # Adding conditions to restrict the hyperparameter space...
        # # ... since learning rate is only used when solver is 'sgd'.
        # use_lr = EqualsCondition(child=learning_rate, parent=solver, value="sgd")
        # # ... since learning rate initialization will only be accounted for when using 'sgd' or 'adam'.
        # use_lr_init = InCondition(child=learning_rate_init, parent=solver, values=["sgd", "adam"])
        # # ... since batch size will not be considered when optimizer is 'lbfgs'.
        # use_batch_size = InCondition(child=batch_size, parent=solver, values=["sgd", "adam"])

        # # We can also add multiple conditions on hyperparameters at once:
        # cs.add_conditions([use_lr, use_batch_size, use_lr_init])

        return cs

    def train(self, config: Configuration, seed: int = 0, budget: int = 25) -> float:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
            parser.add_argument(
                "--type",
                default="linear",
                choices=["linear", "conv"],
                help="model architecture type: "
                + " | ".join(["linear", "conv"])
                + " (default: linear)",
            )
            parser.add_argument(
                "--model",
                default="",
                type=str,
                metavar="MODEL_PATH",
                help="path to model file to load both its architecture and weights (default: none)",
            )
            parser.add_argument(
                "--weights",
                default="",
                type=str,
                metavar="WEIGHTS_PATH",
                help="path to file to load its weights (default: none)",
            )
            parser.add_argument(
                "--shift-depth",
                type=int,
                default=500,
                help="how many layers to convert to shift",
            )
            parser.add_argument(
                "-st",
                "--shift-type",
                default="PS",
                choices=["Q", "PS"],
                help="type of DeepShift method for training and representing weights (default: PS)",
            )
            parser.add_argument(
                "-r",
                "--rounding",
                default="deterministic",
                choices=["deterministic", "stochastic"],
                help="type of rounding (default: deterministic)",
            )
            parser.add_argument(
                "-wb",
                "--weight-bits",
                type=int,
                default=5,
                help="number of bits to represent the shift weights",
            )
            parser.add_argument(
                "-ab",
                "--activation-bits",
                nargs="+",
                default=[16, 16],
                help="number of integer and fraction bits to represent activation (fixed point format)",
            )
            parser.add_argument(
                "-j",
                "--workers",
                default=1,
                type=int,
                metavar="N",
                help="number of data loading workers (default: 1)",
            )
            parser.add_argument(
                "--batch-size",
                type=int,
                default=64,
                metavar="N",
                help="input batch size for training (default: 64)",
            )
            parser.add_argument(
                "--test-batch-size",
                type=int,
                default=1000,
                metavar="N",
                help="input batch size for testing (default: 1000)",
            )
            parser.add_argument(
                "--epochs",
                type=int,
                default=10,
                metavar="N",
                help="number of epochs to train (default: 10)",
            )
            parser.add_argument(
                "-opt",
                "--optimizer",
                metavar="OPT",
                default="SGD",
                help="optimizer algorithm",
            )
            parser.add_argument(
                "--lr",
                type=float,
                default=0.01,
                metavar="LR",
                help="learning rate (default: 0.01)",
            )
            parser.add_argument(
                "--momentum",
                type=float,
                default=0.0,
                metavar="M",
                help="SGD momentum (default: 0.0)",
            )
            parser.add_argument(
                "--resume",
                default="",
                type=str,
                metavar="CHECKPOINT_PATH",
                help="path to latest checkpoint (default: none)",
            )
            parser.add_argument(
                "-e",
                "--evaluate",
                dest="evaluate",
                action="store_true",
                help="only evaluate model on validation set",
            )
            parser.add_argument(
                "--no-cuda",
                action="store_true",
                default=False,
                help="disables CUDA training",
            )
            parser.add_argument(
                "--seed",
                type=int,
                default=1,
                metavar="S",
                help="random seed (default: 1)",
            )
            parser.add_argument(
                "--log-interval",
                type=int,
                default=10,
                metavar="N",
                help="how many batches to wait before logging training status",
            )
            parser.add_argument(
                "--pretrained",
                dest="pretrained",
                default=False,
                type=lambda x: bool(distutils.util.strtobool(x)),
                help="use pre-trained model of full conv or fc model",
            )

            parser.add_argument(
                "--save-model",
                default=True,
                type=lambda x: bool(distutils.util.strtobool(x)),
                help="For Saving the current Model (default: True)",
            )
            parser.add_argument(
                "--print-weights",
                default=True,
                type=lambda x: bool(distutils.util.strtobool(x)),
                help="For printing the weights of Model (default: True)",
            )
            parser.add_argument(
                "--desc",
                type=str,
                default=None,
                help="description to append to model directory name",
            )
            parser.add_argument(
                "--use-kernel",
                type=lambda x: bool(distutils.util.strtobool(x)),
                default=False,
                help="whether using custom shift kernel",
            )
            args = parser.parse_args()
            use_cuda = not args.no_cuda and torch.cuda.is_available()

            args.model_type = config["model_type"]
            args.batch_size = config["batch_size"]
            args.test_batch_size = config["test_batch_size"]
            args.optimizer_type = config["optimizer_type"]
            args.learning_rate = config["learning_rate"]
            args.momentum = config["momentum"]
            args.num_epochs = config["num_epochs"]
            args.weight_bits = config["weight_bits"]
            args.activation_integer_bits = config["activation_integer_bits"]
            args.activation_fraction_bits = config["activation_fraction_bits"]
            args.shift_depth = config["shift_depth"]

            # assert len(args.activation_bits)==2, "activation-bits argument needs to be a tuple of 2 values representing number of integer bits and number of fraction bits, e.g., '3 5' for 8-bits fixed point or '3 13' for 16-bits fixed point"
            # [args.activation_integer_bits, args.activation_fraction_bits] = args.activation_bits
            # [args.activation_integer_bits, args.activation_fraction_bits] = [int(args.activation_integer_bits), int(args.activation_fraction_bits)]

            # if(args.evaluate is False and args.use_kernel is True):
            #     raise ValueError('Our custom kernel currently supports inference only, not training.')
            
            torch.manual_seed(args.seed)

            device = torch.device("cuda" if use_cuda else "cpu")
            kwargs = {'num_workers': args.workers, 'pin_memory': True} if use_cuda else {}
            path_to_local_mnist_data = "data"
            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST(path_to_local_mnist_data, train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                batch_size=args.batch_size, shuffle=True, **kwargs)

            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST(path_to_local_mnist_data, train=False, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,)),
                        ]
                    ),
                ),
                batch_size=args.test_batch_size,
                shuffle=True,
                **kwargs,
            )

            if args.model:
                if args.type or args.pretrained:
                    print(
                        'WARNING: Ignoring arguments "type" and "pretrained" when creating model...'
                    )
                model = None
                saved_checkpoint = torch.load(args.model)
                if isinstance(saved_checkpoint, nn.Module):
                    model = saved_checkpoint
                elif "model" in saved_checkpoint:
                    model = saved_checkpoint["model"]
                else:
                    raise Exception("Unable to load model from " + args.model)
            else:
                if args.type == "linear":
                    model = LinearMNIST().to(device)
                elif args.type == "conv":
                    model = ConvMNIST().to(device)

                if args.pretrained:
                    model.load_state_dict(
                        torch.load(
                            "./models/mnist/simple_"
                            + args.type
                            + "/shift_0/weights.pth"
                        )
                    )
                    model = model.to(device)

            model_rounded = None

            if args.weights:
                saved_weights = torch.load(args.weights)
                if isinstance(saved_weights, nn.Module):
                    state_dict = saved_weights.state_dict()
                elif "state_dict" in saved_weights:
                    state_dict = saved_weights["state_dict"]
                else:
                    state_dict = saved_weights

                model.load_state_dict(state_dict)

            if args.shift_depth > 0:
                model, _ = convert_to_shift(
                    model,
                    args.shift_depth,
                    args.shift_type,
                    convert_all_linear=(args.type != "linear"),
                    convert_weights=True,
                    use_kernel=args.use_kernel,
                    use_cuda=use_cuda,
                    rounding=args.rounding,
                    weight_bits=args.weight_bits,
                    act_integer_bits=args.activation_integer_bits,
                    act_fraction_bits=args.activation_fraction_bits,
                )
                model = model.to(device)
            elif args.use_kernel and args.shift_depth == 0:
                model = convert_to_unoptimized(model)
                model = model.to(device)
            elif args.use_kernel and args.shift_depth == 0:
                model = convert_to_unoptimized(model)
                model = model.to(device)

            loss_fn = F.cross_entropy  # F.nll_loss
            # define optimizer
            optimizer = None
            if args.optimizer.lower() == "sgd":
                optimizer = torch.optim.SGD(
                    model.parameters(), args.lr, momentum=args.momentum
                )
            elif args.optimizer.lower() == "adadelta":
                optimizer = torch.optim.Adadelta(model.parameters(), args.lr)
            elif args.optimizer.lower() == "adagrad":
                optimizer = torch.optim.Adagrad(model.parameters(), args.lr)
            elif args.optimizer.lower() == "adam":
                optimizer = torch.optim.Adam(model.parameters(), args.lr)
            elif args.optimizer.lower() == "rmsprop":
                optimizer = torch.optim.RMSprop(model.parameters(), args.lr)
            elif args.optimizer.lower() == "radam":
                optimizer = optim.RAdam(model.parameters(), args.lr)
            elif args.optimizer.lower() == "ranger":
                optimizer = optim.Ranger(model.parameters(), args.lr)
            else:
                raise ValueError(
                    "Optimizer type: ", args.optimizer, " is not supported or known"
                )

            # optionally resume from a checkpoint
            if args.resume:
                if os.path.isfile(args.resume):
                    print("=> loading checkpoint '{}'".format(args.resume))
                    checkpoint = torch.load(args.resume)
                    if "state_dict" in checkpoint:
                        model.load_state_dict(checkpoint["state_dict"])
                    else:
                        model.load_state_dict(checkpoint)
                    print("=> loaded checkpoint '{}'".format(args.resume))
                else:
                    print("=> no checkpoint found at '{}'".format(args.resume))

            # name model sub-directory "shift_all" if all layers are converted to shift layers
            conv2d_layers_count = count_layer_type(model, nn.Conv2d) + count_layer_type(
                model, unoptimized.UnoptimizedConv2d
            )
            linear_layers_count = count_layer_type(model, nn.Linear) + count_layer_type(
                model, unoptimized.UnoptimizedLinear
            )
            if args.shift_depth > 0:
                if args.shift_type == "Q":
                    shift_label = "shift_q"
                else:
                    shift_label = "shift_ps"
            else:
                shift_label = "shift"

            # name model sub-directory "shift_all" if all layers are converted to shift layers
            conv2d_layers_count = count_layer_type(model, nn.Conv2d)
            linear_layers_count = count_layer_type(model, nn.Linear)
            if conv2d_layers_count == 0 and linear_layers_count == 0:
                shift_label += "_all"
            else:
                shift_label += "_%s" % (args.shift_depth)

            if args.shift_depth > 0:
                shift_label += "_wb_%s" % (args.weight_bits)

            if args.desc is not None and len(args.desc) > 0:
                desc_label = "_%s" % (args.desc)
            else:
                desc_label = ""

            model_name = "simple_%s/%s%s" % (args.type, shift_label, desc_label)

            # if evaluating round weights to ensure that the results are due to powers of 2 weights
            if args.evaluate:
                model = round_shift_weights(model)

            for epoch in range(1, args.num_epochs + 1):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)
                    loss = loss_fn(output, target)
                    loss.backward()
                    optimizer.step()
                    if batch_idx % args.log_interval == 0:
                        print(
                            "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                                epoch,
                                batch_idx * len(data),
                                len(train_loader.dataset),
                                100.0 * batch_idx / len(train_loader),
                                loss.item(),
                            )
                        )

        return loss.item()

    def test(args, model, device, test_loader, loss_fn):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += loss_fn(
                    output, target, reduction="sum"
                ).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print(
            "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

        return test_loss, correct


def plot_trajectory(facades: List[AbstractFacade]) -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")
    plt.ylabel(facades[0].scenario.objectives)
    plt.ylim(0, 0.4)

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            # Single-objective optimization
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=facade.intensifier.__class__.__name__)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()


if __name__ == "__main__":
    dsnn = DSNN()  # Replace with your DSNN class

    facades: List[AbstractFacade] = []
    for intensifier_object in [Hyperband]:
        # Define our environment variables
        scenario = Scenario(
            dsnn.configspace,
            trial_walltime_limit=3000,  # After 60 seconds, we stop the hyperparameter optimization
            n_trials=50,  # Evaluate max 500 different trials
            min_budget=1,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
            max_budget=25,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
            n_workers=1,
        )

        # We want to run five random configurations before starting the optimization.
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

        # Create our intensifier
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

        # Create our SMAC object and pass the scenario and the train method
        smac = MFFacade(
            scenario,
            dsnn.train,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Let's optimize
        incumbent = smac.optimize()

        # Get cost of default configuration
        default_cost = smac.validate(dsnn.configspace.get_default_configuration())
        print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

        # Let's calculate the cost of the incumbent
        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

        facades.append(smac)
        print(incumbent, smac.validate)

    # Let's plot it
    # plot_trajectory(facades)