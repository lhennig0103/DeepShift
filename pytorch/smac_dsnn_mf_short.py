from __future__ import print_function
import warnings
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import optim
from torchvision import datasets, transforms
import os
from deepshift.convert import convert_to_shift, round_shift_weights, count_layer_type
from unoptimized.convert import convert_to_unoptimized
import deepshift
import unoptimized
from smac import HyperparameterOptimizationFacade, Scenario
from smac.intensifier import Hyperband
from smac.facade import MultiFidelityFacade as MFFacade
from smac.facade import AbstractFacade
from ConfigSpace import ConfigurationSpace, Categorical, Integer, Float
import multiprocessing as mp
import matplotlib.pyplot as plt
from typing import List

if mp.get_start_method(allow_none=True) != 'spawn':
    mp.set_start_method('spawn')

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
    def __init__(self, model_config, device):
        self.model_config = model_config
        self.device = device
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        self.loss_fn = F.cross_entropy

    
    def _initialize_model(self):
        # Model initialization based on self.model_config
        if self.model_config['type'] == 'linear':
            model = LinearMNIST().to(self.device)
        elif self.model_config['type'] == 'conv':
            model = ConvMNIST().to(self.device)
        else:
            raise ValueError("Unknown model type specified")

        # Additional model configurations can be added here
        return model
    
    def _initialize_optimizer(self):
        # Optimizer initialization based on self.model_config
        if self.model_config['optimizer'].lower() == 'sgd':
            return torch.optim.SGD(self.model.parameters(), self.model_config['lr'], momentum=self.model_config['momentum'])
        elif self.model_config['optimizer'].lower() == 'adam':
            return torch.optim.Adam(self.model.parameters(), self.model_config['lr'])
        # Add other optimizers as required
        else:
            raise ValueError("Unknown optimizer type specified")
        
    def train(self, train_config):
        # Training configuration setup
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=train_config['batch_size'], shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=train_config['test_batch_size'], shuffle=True)

        for epoch in range(1, train_config['num_epochs'] + 1):
            self.model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % train_config['log_interval'] == 0:
                    print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                          f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

        test_loss, test_accuracy = self.test(self.model, self.device, test_loader)
        return test_loss

    def test(self, model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.cross_entropy(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({test_accuracy:.0f}%)')
        return test_loss, test_accuracy
    

def create_configspace():
    cs = ConfigurationSpace()

    model_type = Categorical("model_type", ["linear", "conv"])
    batch_size = Integer("batch_size", (32, 128))
    test_batch_size = Integer("test_batch_size", (500, 2000))
    optimizer_type = Categorical("optimizer_type", ["SGD", "Adam"])
    learning_rate = Float("learning_rate", (0.001, 0.1))
    momentum = Float("momentum", (0.0, 0.9))
    num_epochs = Integer("num_epochs", (5, 20))
    weight_bits = Integer("weight_bits", (2, 8))
    activation_integer_bits = Integer("activation_integer_bits", (2, 32))
    activation_fraction_bits = Integer("activation_fraction_bits", (2, 32))
    shift_depth = Integer("shift_depth", (0, 1500))

    cs.add_hyperparameters([model_type, batch_size, test_batch_size, optimizer_type,
                            learning_rate, momentum, num_epochs, weight_bits,
                            activation_integer_bits, activation_fraction_bits, shift_depth])

    return cs

def train_model(config, seed: int = 0, budget: int = 25):
    model_config = {
        'type': config['model_type'],
        'optimizer': config['optimizer_type'],
        'lr': config['learning_rate'],
        'momentum': config['momentum'],
        'weight_bits': config['weight_bits'],
        'activation_integer_bits': config['activation_integer_bits'],
        'activation_fraction_bits': config['activation_fraction_bits'],
        'shift_depth': config['shift_depth']
    }

    train_config = {
        'batch_size': config['batch_size'],
        'test_batch_size': config['test_batch_size'],
        'num_epochs': min(config['num_epochs'], int(budget)),  # Use budget to limit epochs
        'log_interval': 10  # You can adjust this as needed
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    torch.manual_seed(seed)  # Set the random seed for reproducibility
    
    dsnn = DSNN(model_config, device)
    loss = dsnn.train(train_config)

    return loss


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

def main():
    
    cs = create_configspace()
    facades: list[AbstractFacade] = []
    scenario = Scenario(
            cs,
            trial_walltime_limit=3000,  # After 60 seconds, we stop the hyperparameter optimization
            n_trials=50,  # Evaluate max 500 different trials
            min_budget=1,  # Train the MLP using a hyperparameter configuration for at least 5 epochs
            max_budget=25,  # Train the MLP using a hyperparameter configuration for at most 25 epochs
            n_workers=1,
        )
    
    # Create our intensifier
    intensifier_object = Hyperband
    intensifier = intensifier_object(scenario, incumbent_selection="highest_budget")
    
    # We want to run five random configurations before starting the optimization.
    initial_design = MFFacade.get_initial_design(scenario, n_configs=5)

    smac = MFFacade(
            scenario,
            train_model,
            initial_design=initial_design,
            intensifier=intensifier,
            overwrite=True,
        )
    incumbent = smac.optimize()

    # Get cost of default configuration
    default_cost = smac.validate(cs.get_default_configuration())
    print(f"Default cost ({intensifier.__class__.__name__}): {default_cost}")

    # Let's calculate the cost of the incumbent
    incumbent_cost = smac.validate(incumbent)
    print(f"Incumbent cost ({intensifier.__class__.__name__}): {incumbent_cost}")

    facades.append(smac)
    
    print(incumbent, smac.validate)
    plot_trajectory(facades)

if __name__ == "__main__":
    main()
