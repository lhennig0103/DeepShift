#DSNN imports
from __future__ import print_function
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

import deepshift
import unoptimized
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from unoptimized.convert import convert_to_unoptimized

#SMAC imports

import warnings
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from ConfigSpace import (
    Categorical,
    Configuration,
    ConfigurationSpace,
    EqualsCondition,
    Float,
    InCondition,
    Integer,
)
from sklearn.datasets import load_digits
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from smac import MultiFidelityFacade as MFFacade
from smac import Scenario
from smac.facade import AbstractFacade
from smac.intensifier.hyperband import Hyperband
from smac.intensifier.successive_halving import SuccessiveHalving

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ])),
    batch_size=test_batch_size, shuffle=False)

class LinearMNIST(nn.Module):
    def __init__(self):
        super(LinearMNIST, self).__init__()
        self.fc1 = nn.Linear(1*28*28, 512)
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
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class DSNN:
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
        activation_integer_bits = Integer("activation_integer_bits", (8, 16), default=16)
        activation_fraction_bits = Integer("activation_fraction_bits", (8, 16), default=16)



        # Add all hyperparameters at once:
        cs.add_hyperparameters([
            model_type,
            batch_size,
            test_batch_size,
            optimizer_type,
            learning_rate,
            momentum,
            num_epochs,
            weight_bits,
            activation_integer_bits,
            activation_fraction_bits
        ])

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

            model_type = config["model_type"]
            batch_size = config["batch_size"]
            test_batch_size = config["test_batch_size"]
            optimizer_type = config["optimizer_type"]
            learning_rate = config["learning_rate"]
            momentum = config["momentum"]
            num_epochs = config["num_epochs"]
            weight_bits = config["weight_bits"]
            activation_integer_bits = config["activation_integer_bits"]
            activation_fraction_bits = config["activation_fraction_bits"]

            use_cuda = torch.cuda.is_available()
            device = torch.device("cuda" if use_cuda else "cpu")

            train_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(
                datasets.MNIST('../data', train=False,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))
                            ])),
                batch_size=test_batch_size, shuffle=False)

            if model_type == 'linear':
                model = LinearMNIST().to(device)
            elif model_type == 'conv':
                model = ConvMNIST().to(device)

            if optimizer_type == "SGD":
                optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum)
            
            if optimizer_type == "Adam":
                optimizer = torch.optim.Adam(model.parameters(), learning_rate, momentum=momentum)

            for epoch in range(1, num_epochs + 1):
                dsnn.train(model, device, train_loader, optimizer, epoch)
        test_loss, accuracy = test(model, device, test_loader)
        return 1 - accuracy

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
            dsnn.configspace,  # Make sure this is defined in your DSNN class
            trial_walltime_limit=6000,
            n_trials=200,
            min_budget=3,
            max_budget=100,
            n_workers=1,
        )

        # Initial design setup remains the same
        initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

        # Intensifier setup remains the same
        intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")

        # Create the SMAC object, passing in the scenario and the DSNN train method
        smac = MFFacade(
            scenario,
            dsnn.train,  # The train method should be defined in your DSNN class
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )

        # Optimization process remains the same
        incumbent = smac.optimize()

        # Validation process remains the same
        default_cost = smac.validate(dsnn.configspace.get_default_configuration())
        print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

        incumbent_cost = smac.validate(incumbent)
        print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

        facades.append(smac)

    # Plotting remains the same
    plot_trajectory(facades)

    # After running the optimization
    print("Best configuration found:", incumbent)
