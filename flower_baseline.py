import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.profiler import profile, ProfilerActivity
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset

import flwr as fl
import ray
from flwr.common import (
    Context,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
import argparse
import numpy as np
import os
import csv
import time
from dataclasses import dataclass
from collections import OrderedDict
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import logging
import urllib.request
import json
import base64
import warnings
import random

# Suppress warnings
logging.getLogger("flwr").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*mode.*deprecated.*Pillow.*")
warnings.filterwarnings("ignore", message=".*urllib3.*OpenSSL.*")

from captum.attr import visualization as vit
from newAux import FedGSW
from strategy_laa import FedLAA
from strategy_ama import FedAMA
from strategy_de import FedDE
from strategy_sbfl import FedSBFL
from strategy_sequential import SequentialRoundRobin

DATASET_HYPERPARAMS = {
    'mnist': {'lr': 0.01, 'batch_size': 32, 'epochs': 1},
    'cifar10': {'lr': 0.01, 'batch_size': 64, 'epochs': 1},
    'stl10': {'lr': 0.01, 'batch_size': 32, 'epochs': 1},
    'oxfordpet': {'lr': 0.005, 'batch_size': 16, 'epochs': 1},
    'adult': {'lr': 0.01, 'batch_size': 128, 'epochs': 1},
    'speechcommands': {'lr': 0.01, 'batch_size': 32, 'epochs': 1}
}

LEARNING_TYPE_ALIASES = {
    'CL': 'CL',
    'CENTRALIZED': 'CL',
    'CENTRALIZED_LEARNING': 'CL',
    'FL': 'FL',
    'FEDERATED': 'FL',
    'FEDERATED_LEARNING': 'FL',
    'SL': 'SL',
    'SPLIT': 'SL',
    'SPLIT_LEARNING': 'SL',
    'SFLV1': 'SFLV1',
    'SPLITFEDV1': 'SFLV1',
    'SPLITFED_V1': 'SFLV1',
    'SPLITFED_V_1': 'SFLV1',
    'SPLITFED_LEARNING_V1': 'SFLV1',
    'SPLITFED_LEARNING_V_1': 'SFLV1',
    'SFL_V1': 'SFLV1',
    'SFL_V_1': 'SFLV1',
    'SFLV2': 'SFLV2',
    'SPLITFEDV2': 'SFLV2',
    'SPLITFED_V2': 'SFLV2',
    'SPLITFED_V_2': 'SFLV2',
    'SPLITFED_LEARNING_V2': 'SFLV2',
    'SPLITFED_LEARNING_V_2': 'SFLV2',
    'SFL_V2': 'SFLV2',
    'SFL_V_2': 'SFLV2',
    'CFL': 'CFL',
    'CONTINUAL_FEDERATED': 'CFL',
    'CONTINUAL_FEDERATED_LEARNING': 'CFL',
    'CFSL': 'CFSL',
    'CONTINUAL_FEDERATED_SPLIT': 'CFSL',
    'CONTINUAL_FEDERATED_SPLIT_LEARNING': 'CFSL',
}

LEARNING_TYPE_DISPLAY_NAMES = {
    'CL': 'Centralized Learning',
    'FL': 'Federated Learning',
    'SL': 'Split Learning',
    'SFLV1': 'SplitFed Learning v1',
    'SFLV2': 'SplitFed Learning v2',
    'CFL': 'Continual Federated Learning',
    'CFSL': 'Continual Federated Split Learning',
}

LEARNING_TYPE_FILE_TAGS = {
    'CL': 'cl',
    'FL': 'fl',
    'SL': 'sl',
    'SFLV1': 'sflv1',
    'SFLV2': 'sflv2',
    'CFL': 'cfl',
    'CFSL': 'cfsl',
}

TRAIN_FLOP_MULTIPLIER = 3.0

class CNN(nn.Module):
    def __init__(self, num_classes=10, input_size=128, in_channels=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1) 
        self.pool = nn.MaxPool2d(2, 2)
        
        dummy_input = torch.zeros(1, in_channels, input_size, input_size)
        x = self.pool(F.relu(self.conv1(dummy_input)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        self.fc_input_dim = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(self.fc_input_dim, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, self.fc_input_dim)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, num_classes=2, hidden_dims=[64, 32]):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return x.squeeze(2)

def get_model(name, num_classes=10, input_size=128, in_channels=3, input_dim=None):
    if name.lower() == 'cnn':
        return CNN(num_classes=num_classes, input_size=input_size, in_channels=in_channels)
    
    elif name.lower() == 'mlp':
        if input_dim is None:
            raise ValueError("input_dim must be specified for MLP model")
        return MLP(input_dim=input_dim, num_classes=num_classes)

    elif name.lower() == 'm5':
        return M5(n_input=in_channels, n_output=num_classes)
    
    elif name.lower() == 'squeezenet':
        model = models.squeezenet1_1(num_classes=num_classes)
        if in_channels != 3:
            model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, stride=2)
        return model
    
    elif name.lower() == 'shufflenet':
        model = models.shufflenet_v2_x0_5(num_classes=num_classes)
        if in_channels != 3:
            model.conv1[0] = nn.Conv2d(in_channels, 24, kernel_size=3, stride=2, padding=1, bias=False)
        return model
    
    elif name.lower() == 'resnet':
        model = models.resnet18(num_classes=num_classes)
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        return model
    
    else:
        raise ValueError(f"Unsupported model: {name}")

def get_split_models(name, num_classes=10, input_size=128, in_channels=3, input_dim=None):
    if name.lower() == 'cnn':
        class CNNFront(nn.Module):
            def __init__(self, in_channels):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
            def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                return x
        
        class CNNBack(nn.Module):
            def __init__(self, num_classes, input_size, in_channels):
                super().__init__()
                self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                dummy_input = torch.zeros(1, in_channels, input_size, input_size)
                dummy_front = CNNFront(in_channels)
                with torch.no_grad():
                    x = self.pool(F.relu(self.conv3(dummy_front(dummy_input))))
                self.fc_input_dim = x.view(1, -1).size(1)
                self.fc1 = nn.Linear(self.fc_input_dim, 64)
                self.fc2 = nn.Linear(64, num_classes)
            def forward(self, x):
                x = self.pool(F.relu(self.conv3(x)))
                x = x.view(-1, self.fc_input_dim)
                x = F.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        return CNNFront(in_channels), CNNBack(num_classes, input_size, in_channels)

    elif name.lower() == 'mlp':
        class MLPFront(nn.Module):
            def __init__(self, input_dim):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2))
            def forward(self, x):
                return self.net(x)
        class MLPBack(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2), nn.Linear(32, num_classes))
            def forward(self, x):
                return self.net(x)
        return MLPFront(input_dim), MLPBack(num_classes)

    elif name.lower() == 'm5':
        class M5Front(nn.Module):
            def __init__(self, n_input, stride=16, n_channel=32):
                super().__init__()
                self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
                self.bn1 = nn.BatchNorm1d(n_channel)
                self.pool1 = nn.MaxPool1d(4)
                self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
                self.bn2 = nn.BatchNorm1d(n_channel)
                self.pool2 = nn.MaxPool1d(4)
            def forward(self, x):
                x = self.pool1(F.relu(self.bn1(self.conv1(x))))
                x = self.pool2(F.relu(self.bn2(self.conv2(x))))
                return x
        class M5Back(nn.Module):
            def __init__(self, n_output, n_channel=32):
                super().__init__()
                self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
                self.bn3 = nn.BatchNorm1d(2 * n_channel)
                self.pool3 = nn.MaxPool1d(4)
                self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
                self.bn4 = nn.BatchNorm1d(2 * n_channel)
                self.pool4 = nn.MaxPool1d(4)
                self.fc1 = nn.Linear(2 * n_channel, n_output)
            def forward(self, x):
                x = self.pool3(F.relu(self.bn3(self.conv3(x))))
                x = self.pool4(F.relu(self.bn4(self.conv4(x))))
                x = F.avg_pool1d(x, x.shape[-1]).permute(0, 2, 1)
                return self.fc1(x).squeeze(2)
        return M5Front(in_channels), M5Back(num_classes)

    else:
        full_model = get_model(name, num_classes, input_size, in_channels, input_dim)
        if name.lower() == 'squeezenet':
            front = nn.Sequential(*list(full_model.features.children())[:5])
            class SqueezeNetBack(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.features_back = nn.Sequential(*list(model.features.children())[5:])
                    self.classifier = model.classifier
                def forward(self, x):
                    x = self.features_back(x)
                    x = self.classifier(x)
                    return torch.flatten(x, 1)
            return front, SqueezeNetBack(full_model)
        elif name.lower() == 'shufflenet':
            front = nn.Sequential(full_model.conv1, full_model.maxpool, full_model.stage2)
            class ShuffleNetBack(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.stage3 = model.stage3
                    self.stage4 = model.stage4
                    self.conv5 = model.conv5
                    self.fc = model.fc
                def forward(self, x):
                    x = self.stage3(x)
                    x = self.stage4(x)
                    x = self.conv5(x)
                    x = x.mean([2, 3])
                    return self.fc(x)
            return front, ShuffleNetBack(full_model)
        elif name.lower() == 'resnet':
            front = nn.Sequential(full_model.conv1, full_model.bn1, full_model.relu, full_model.maxpool, full_model.layer1)
            class ResNetBack(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.layer2 = model.layer2
                    self.layer3 = model.layer3
                    self.layer4 = model.layer4
                    self.avgpool = model.avgpool
                    self.fc = model.fc
                def forward(self, x):
                    x = self.layer2(x)
                    x = self.layer3(x)
                    x = self.layer4(x)
                    x = self.avgpool(x)
                    x = torch.flatten(x, 1)
                    return self.fc(x)
            return front, ResNetBack(full_model)
        else:
            raise ValueError(f"Split implementation not found for {name}")


def build_sample_input(input_dim=None, in_channels=None, img_size=None):
    if input_dim is not None:
        return torch.randn(1, input_dim)
    if in_channels is None or img_size is None:
        raise ValueError("Either input_dim or both in_channels/img_size must be provided")
    if isinstance(img_size, tuple):
        return torch.randn(1, in_channels, *img_size)
    return torch.randn(1, in_channels, img_size, img_size) if img_size != 8000 else torch.randn(1, in_channels, img_size)


def estimate_forward_flops(model: nn.Module, sample_input: torch.Tensor) -> float:
    """Estimate forward FLOPs on CPU using PyTorch profiler."""
    cpu_model = model.cpu().eval()
    cpu_input = sample_input.detach().cpu()
    try:
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            with torch.no_grad():
                cpu_model(cpu_input)
        return float(prof.key_averages().total_average().flops)
    except Exception as exc:
        print(f"  - Warning: FLOPs estimation failed for {cpu_model.__class__.__name__}: {exc}")
        return 0.0


def estimate_model_flops(
    model_name: str,
    num_classes: int,
    is_split: bool,
    input_dim=None,
    in_channels=None,
    img_size=None,
) -> float:
    sample_input = build_sample_input(input_dim=input_dim, in_channels=in_channels, img_size=img_size)
    if is_split:
        front_model, back_model = get_split_models(
            model_name,
            num_classes=num_classes,
            input_size=img_size if input_dim is None else 128,
            in_channels=in_channels if in_channels is not None else 3,
            input_dim=input_dim,
        )
        front_flops = estimate_forward_flops(front_model, sample_input)
        with torch.no_grad():
            smashed_sample = front_model.cpu().eval()(sample_input.cpu())
        back_flops = estimate_forward_flops(back_model, smashed_sample)
        return front_flops + back_flops

    model = get_model(
        model_name,
        num_classes=num_classes,
        input_size=img_size if input_dim is None else 128,
        in_channels=in_channels if in_channels is not None else 3,
        input_dim=input_dim,
    )
    return estimate_forward_flops(model, sample_input)

def sync_device(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def model_state_to_numpy(model: nn.Module) -> List[np.ndarray]:
    return [val.detach().cpu().numpy() for _, val in model.state_dict().items()]


def load_numpy_state(model: nn.Module, parameters: List[np.ndarray]) -> None:
    state_dict = OrderedDict(
        {k: torch.tensor(v) for k, v in zip(model.state_dict().keys(), parameters)}
    )
    model.load_state_dict(state_dict, strict=True)


def aggregate_parameter_lists(
    parameters_list: List[List[np.ndarray]],
    weights: List[float],
) -> List[np.ndarray]:
    if not parameters_list:
        return []
    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        total_weight = float(len(weights))
        weights = [1.0 for _ in weights]

    aggregated: List[np.ndarray] = []
    num_layers = len(parameters_list[0])
    normalized_weights = [float(weight) / total_weight for weight in weights]
    for layer_idx in range(num_layers):
        layer_sum = None
        for client_idx, params in enumerate(parameters_list):
            contribution = params[layer_idx] * normalized_weights[client_idx]
            layer_sum = contribution if layer_sum is None else layer_sum + contribution
        aggregated.append(layer_sum)
    return aggregated


def build_split_back_model(
    model_name: str,
    num_classes: int,
    input_dim=None,
    in_channels=None,
    img_size=None,
) -> nn.Module:
    if input_dim is not None:
        _, back_model = get_split_models(model_name, num_classes=num_classes, input_dim=input_dim)
    elif img_size == 8000:
        _, back_model = get_split_models(model_name, num_classes=num_classes, in_channels=in_channels)
    else:
        _, back_model = get_split_models(
            model_name,
            num_classes=num_classes,
            input_size=img_size,
            in_channels=in_channels,
        )
    return back_model


@ray.remote
class SplitFedServerCopy:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float,
        device: str,
        input_dim=None,
        in_channels=None,
        img_size=None,
    ):
        self.model = build_split_back_model(
            model_name=model_name,
            num_classes=num_classes,
            input_dim=input_dim,
            in_channels=in_channels,
            img_size=img_size,
        ).to(device)
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        load_numpy_state(self.model, parameters)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def get_parameters(self) -> List[np.ndarray]:
        return model_state_to_numpy(self.model)

    def train_batch(self, smashed_data: np.ndarray, labels: np.ndarray) -> Dict[str, Union[float, np.ndarray]]:
        start_time = time.perf_counter()
        self.model.train()
        smashed = torch.tensor(smashed_data, dtype=torch.float32, device=self.device, requires_grad=True)
        target = torch.tensor(labels, dtype=torch.long, device=self.device)
        self.optimizer.zero_grad()
        output = self.model(smashed)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        sync_device(self.device)
        server_compute_time_sec = time.perf_counter() - start_time
        return {
            "status": "ok",
            "smashed_grad": smashed.grad.detach().cpu().numpy(),
            "loss": float(loss.item()),
            "server_compute_time_sec": float(server_compute_time_sec),
        }

    def evaluate_batch(self, smashed_data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        start_time = time.perf_counter()
        self.model.eval()
        smashed = torch.tensor(smashed_data, dtype=torch.float32, device=self.device)
        target = torch.tensor(labels, dtype=torch.long, device=self.device)
        with torch.no_grad():
            output = self.model(smashed)
            loss = self.criterion(output, target)
            predicted = output.argmax(dim=1)
            correct = int((predicted == target).sum().item())
            total = int(target.size(0))
        sync_device(self.device)
        server_compute_time_sec = time.perf_counter() - start_time
        return {
            "loss_sum": float(loss.item() * total),
            "correct": float(correct),
            "total": float(total),
            "server_compute_time_sec": float(server_compute_time_sec),
        }


@ray.remote
class SplitFedV2Server:
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        lr: float,
        device: str,
        input_dim=None,
        in_channels=None,
        img_size=None,
    ):
        self.model = build_split_back_model(
            model_name=model_name,
            num_classes=num_classes,
            input_dim=input_dim,
            in_channels=in_channels,
            img_size=img_size,
        ).to(device)
        self.device = device
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        self.current_round = 0
        self.expected_turn = 0
        self.num_expected_clients = 0

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        load_numpy_state(self.model, parameters)
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def start_round(self, server_round: int, num_expected_clients: int) -> None:
        self.current_round = int(server_round)
        self.expected_turn = 0
        self.num_expected_clients = max(0, int(num_expected_clients))
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

    def get_parameters(self) -> List[np.ndarray]:
        return model_state_to_numpy(self.model)

    def train_batch(
        self,
        server_round: int,
        client_turn: int,
        smashed_data: np.ndarray,
        labels: np.ndarray,
    ) -> Dict[str, Union[str, float, np.ndarray]]:
        if int(server_round) != self.current_round:
            return {"status": "stale"}
        if self.expected_turn >= self.num_expected_clients:
            return {"status": "done"}
        if int(client_turn) != self.expected_turn:
            return {"status": "wait"}

        start_time = time.perf_counter()
        self.model.train()
        smashed = torch.tensor(smashed_data, dtype=torch.float32, device=self.device, requires_grad=True)
        target = torch.tensor(labels, dtype=torch.long, device=self.device)
        self.optimizer.zero_grad()
        output = self.model(smashed)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        sync_device(self.device)
        server_compute_time_sec = time.perf_counter() - start_time
        return {
            "status": "ok",
            "smashed_grad": smashed.grad.detach().cpu().numpy(),
            "loss": float(loss.item()),
            "server_compute_time_sec": float(server_compute_time_sec),
        }

    def finish_client(self, server_round: int, client_turn: int) -> Dict[str, str]:
        if int(server_round) != self.current_round:
            return {"status": "stale"}
        if self.expected_turn >= self.num_expected_clients:
            return {"status": "done"}
        if int(client_turn) != self.expected_turn:
            return {"status": "wait"}
        self.expected_turn += 1
        return {"status": "ok"}

    def evaluate_batch(self, smashed_data: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        start_time = time.perf_counter()
        self.model.eval()
        smashed = torch.tensor(smashed_data, dtype=torch.float32, device=self.device)
        target = torch.tensor(labels, dtype=torch.long, device=self.device)
        with torch.no_grad():
            output = self.model(smashed)
            loss = self.criterion(output, target)
            predicted = output.argmax(dim=1)
            correct = int((predicted == target).sum().item())
            total = int(target.size(0))
        sync_device(self.device)
        server_compute_time_sec = time.perf_counter() - start_time
        return {
            "loss_sum": float(loss.item() * total),
            "correct": float(correct),
            "total": float(total),
            "server_compute_time_sec": float(server_compute_time_sec),
        }


class SplitFedV1Strategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args,
        server_actor_names: List[str],
        server_actor_kwargs: Dict[str, Union[str, int, float, None]],
        server_parameters: List[np.ndarray],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_actor_names = list(server_actor_names)
        self.server_actor_kwargs = dict(server_actor_kwargs)
        self.server_actors: List["ray.actor.ActorHandle"] = []
        self.server_parameters = server_parameters
        self.round_actor_slots: Dict[str, int] = {}

    def _ensure_server_actors(self) -> None:
        if self.server_actors:
            return
        self.server_actors = [
            SplitFedServerCopy.options(name=actor_name).remote(**self.server_actor_kwargs)
            for actor_name in self.server_actor_names
        ]
        ray.get([actor.set_parameters.remote(self.server_parameters) for actor in self.server_actors])

    def configure_fit(self, server_round, parameters, client_manager):
        self._ensure_server_actors()
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)
        if not fit_configurations:
            return fit_configurations
        self.round_actor_slots = {}
        actor_calls = []
        updated_fit_configurations = []
        for slot, (client_proxy, fit_ins) in enumerate(fit_configurations):
            self.round_actor_slots[str(client_proxy.cid)] = slot
            actor_calls.append(self.server_actors[slot].set_parameters.remote(self.server_parameters))
            fit_config = dict(fit_ins.config)
            fit_config["sfl_server_slot"] = int(slot)
            updated_fit_configurations.append((client_proxy, FitIns(fit_ins.parameters, fit_config)))
        ray.get(actor_calls)
        return updated_fit_configurations

    def aggregate_fit(self, server_round, results, failures):
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        if not results:
            return aggregated_parameters, metrics

        selected_slots = [self.round_actor_slots[str(client_proxy.cid)] for client_proxy, _ in results]
        weights = [fit_res.num_examples for _, fit_res in results]
        server_params = ray.get([self.server_actors[slot].get_parameters.remote() for slot in selected_slots])
        self.server_parameters = aggregate_parameter_lists(server_params, weights)
        ray.get([actor.set_parameters.remote(self.server_parameters) for actor in self.server_actors])
        self.round_actor_slots = {}
        return aggregated_parameters, metrics


class SplitFedV2Strategy(fl.server.strategy.FedAvg):
    def __init__(
        self,
        *args,
        server_actor_name: str,
        server_actor_kwargs: Dict[str, Union[str, int, float, None]],
        server_parameters: List[np.ndarray],
        seed: int = 42,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.server_actor_name = str(server_actor_name)
        self.server_actor_kwargs = dict(server_actor_kwargs)
        self.server_parameters = server_parameters
        self.server_actor = None
        self.seed = int(seed)

    def _ensure_server_actor(self):
        if self.server_actor is not None:
            return
        self.server_actor = SplitFedV2Server.options(name=self.server_actor_name).remote(**self.server_actor_kwargs)
        ray.get(self.server_actor.set_parameters.remote(self.server_parameters))

    def configure_fit(self, server_round, parameters, client_manager):
        self._ensure_server_actor()
        fit_configurations = super().configure_fit(server_round, parameters, client_manager)
        if not fit_configurations:
            return fit_configurations
        turn_order = list(range(len(fit_configurations)))
        rng = random.Random(self.seed + int(server_round))
        rng.shuffle(turn_order)
        updated_fit_configurations = []
        for turn, (client_proxy, fit_ins) in zip(turn_order, fit_configurations):
            fit_config = dict(fit_ins.config)
            fit_config["sfl_v2_turn"] = int(turn)
            updated_fit_configurations.append((client_proxy, FitIns(fit_ins.parameters, fit_config)))
        ray.get(self.server_actor.start_round.remote(server_round, len(fit_configurations)))
        return updated_fit_configurations

def warmup(model_name, num_classes, img_size, in_channels, device):
    """Performs a dummy forward-backward pass to initialize hardware and libraries."""
    print("  - Warming up hardware (first-run initialization)...")
    model = get_model(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels).to(device)
    dummy_input = torch.randn(1, in_channels, img_size, img_size).to(device)
    model.train()
    try:
        output = model(dummy_input)
        loss = output.sum()
        loss.backward()
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
    except Exception as e:
        print(f"    Warning: Warmup failed (not critical): {e}")
    print("  - Warmup complete.")

# --- 3. Dataset Loader ---

class TabularDataset(torch.utils.data.Dataset):
    """Custom dataset for tabular data."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_dataset(name, img_size):
    if name.lower() == 'adult':
        from sklearn.datasets import fetch_openml
        from sklearn.preprocessing import StandardScaler, LabelEncoder
        from sklearn.model_selection import train_test_split
        
        print("  - Downloading Adult Income dataset...")
        data = fetch_openml('adult', version=2, as_frame=True, parser='auto')
        X = data.data.copy()
        y = data.target.copy()
        
        for col in X.select_dtypes(include=['category', 'object']).columns:
            X[col] = X[col].astype(str)
            X.loc[:, col] = LabelEncoder().fit_transform(X[col])
        
        y = LabelEncoder().fit_transform(y)
        
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        train_set = TabularDataset(X_train, y_train)
        test_set = TabularDataset(X_test, y_test)
        targets = y_train
        
        num_classes = len(np.unique(y_train)) # Dynamically determine num_classes for adult
        
        return train_set, test_set, targets, num_classes
    
    elif name.lower() == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
        num_classes = 10
    elif name.lower() == 'stl10':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        train_set = datasets.STL10(root='./data', split='train', download=True, transform=transform)
        test_set = datasets.STL10(root='./data', split='test', download=True, transform=transform)
        targets = np.array(train_set.labels)
        num_classes = 10
    elif name.lower() == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
        targets = np.array(train_set.targets)
        num_classes = 10
    elif name.lower() == 'oxfordpet':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        train_set = datasets.OxfordIIITPet(root='./data', split='trainval', download=True, transform=transform)
        test_set = datasets.OxfordIIITPet(root='./data', split='test', download=True, transform=transform)
        targets = np.array(train_set._labels)
        num_classes = 37
    elif name.lower() == 'speechcommands':
        import torchaudio
        class SubsetSC(torchaudio.datasets.SPEECHCOMMANDS):

            def __init__(self, subset: str = None):
                super().__init__("./data", download=True)
                
                def load_list(filename):
                    filepath = os.path.join(self._path, filename)
                    with open(filepath) as fileobj:
                        return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]
                
                if subset == "validation":
                    self._walker = load_list("validation_list.txt")
                elif subset == "testing":
                    self._walker = load_list("testing_list.txt")
                elif subset == "training":
                    excludes = load_list("validation_list.txt") + load_list("testing_list.txt")
                    excludes = set(excludes)
                    self._walker = [w for w in self._walker if w not in excludes]
        
        if name.lower() == 'speechcommands':
            # Use 4-class subset as requested by user
            # Common commands: 'yes', 'no', 'up', 'down'
            subset_labels = ['yes', 'no', 'up', 'down']
            label_to_index = {label: index for index, label in enumerate(subset_labels)}
            num_classes = len(subset_labels)
            
            # Helper to check if a sample is in the subset
            def is_in_subset(datapoint):
                return datapoint[2] in subset_labels

            # Load and filter raw datasets
            train_set_raw = [d for d in SubsetSC("training") if is_in_subset(d)]
            test_set_raw = [d for d in SubsetSC("testing") if is_in_subset(d)]
            
            if not train_set_raw:
                # Fallback to standard 10+2 if subset empty (shouldn't happen with standard dataset)
                print("Warning: Requested subset empty. Using full dataset.")
                all_labels = sorted(list(set(datapoint[2] for datapoint in SubsetSC("training"))))
                labels = [l for l in all_labels if l != "_background_noise_"]
                label_to_index = {label: index for index, label in enumerate(labels)}
                num_classes = len(labels)
                train_set_raw = [d for d in SubsetSC("training") if d[2] != "_background_noise_"]
                test_set_raw = [d for d in SubsetSC("testing") if d[2] != "_background_noise_"]


        class SPEECHCOMMANDS_Processed(torch.utils.data.Dataset):
            def __init__(self, dataset, label_to_index):
                self.dataset = dataset
                self.label_to_index = label_to_index
                self.new_sample_rate = 8000
                self.transform = torchaudio.transforms.Resample(orig_freq=16000, new_freq=self.new_sample_rate)

            def __getitem__(self, n):
                waveform, sample_rate, label, _, _ = self.dataset[n]
                # Filter out background_noise (often has different shape or no label) if necessary, 
                # but standard SubsetSC usually handles valid files. 
                # Note: M5 expects 1xInputLength. Pad to 1 sec (8000 samples).
                
                waveform = self.transform(waveform)
                
                # Pad or truncate to 1 second
                if waveform.shape[-1] < self.new_sample_rate:
                    waveform = torch.nn.functional.pad(waveform, (0, self.new_sample_rate - waveform.shape[-1]))
                else:
                    waveform = waveform[:, :self.new_sample_rate]
                
                label_idx = self.label_to_index.get(label, 0)
                
                return waveform, torch.tensor(label_idx, dtype=torch.long)


            def __len__(self):
                return len(self.dataset)

        train_set = SPEECHCOMMANDS_Processed(train_set_raw, label_to_index)
        test_set = SPEECHCOMMANDS_Processed(test_set_raw, label_to_index)
        
        # Extract targets for partitioning
        print(f"  - Processing {num_classes}-class SpeechCommands targets...")
        targets = []
        for i in range(len(train_set)):
            _, label = train_set[i]
            # Ensure label is int for np.array
            targets.append(int(label))
        targets = np.array(targets, dtype=np.int64)
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return train_set, test_set, targets, num_classes

def partition_data(dataset, targets, num_clients, alpha, num_classes):
    indices = [[] for _ in range(num_clients)]
    
    if alpha >= 1.0:
        all_indices = np.arange(len(dataset))
        np.random.shuffle(all_indices)
        indices = [idx.tolist() for idx in np.array_split(all_indices, num_clients)]
    else:
        for k in range(num_classes):
            idx_k = np.where(targets == k)[0]
            if len(idx_k) == 0: continue
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            
            idx_splits = np.split(idx_k, proportions)
            for i in range(num_clients):
                indices[i].extend(idx_splits[i].tolist())
                
        for i in range(num_clients):
            np.random.shuffle(indices[i])

    empty_clients = [i for i, idxs in enumerate(indices) if len(idxs) == 0]
    for empty_idx in empty_clients:
        donor_idx = int(np.argmax([len(idxs) for idxs in indices]))
        if len(indices[donor_idx]) <= 1:
            continue
        indices[empty_idx].append(indices[donor_idx].pop())

    return indices


def normalize_learning_type(raw_value: str) -> str:
    normalized = str(raw_value).strip().upper().replace('-', '_').replace(' ', '_')
    if normalized not in LEARNING_TYPE_ALIASES:
        valid = ", ".join(LEARNING_TYPE_DISPLAY_NAMES.keys())
        raise ValueError(f"Unsupported learning type '{raw_value}'. Valid values: {valid}")
    return LEARNING_TYPE_ALIASES[normalized]


@dataclass
class ContinualRoundSelection:
    indices: List[int]
    current_examples: int
    replay_examples: int
    unique_examples: int


@dataclass
class ContinualSchedule:
    experiences: List[List[int]]
    replay_ratio: float = 0.25
    seed: int = 42
    budget_mode: str = "experience"
    sample_with_replacement: bool = True

    @property
    def partition_size(self) -> int:
        return int(sum(len(exp) for exp in self.experiences))

    def _target_budget(self, experience_idx: int, current_size: int) -> int:
        if self.budget_mode == "partition":
            return self.partition_size
        if self.budget_mode == "seen":
            seen_so_far = sum(len(exp) for exp in self.experiences[:experience_idx + 1])
            return min(seen_so_far, current_size + int(round(current_size * self.replay_ratio)))
        return current_size + int(round(current_size * self.replay_ratio))

    def indices_for_round(self, server_round: int, client_id: int) -> ContinualRoundSelection:
        if not self.experiences:
            return ContinualRoundSelection(indices=[], current_examples=0, replay_examples=0, unique_examples=0)

        experience_idx = min(max(server_round, 1) - 1, len(self.experiences) - 1)
        active_indices = list(self.experiences[experience_idx])
        current_examples = len(active_indices)
        target_budget = max(current_examples, self._target_budget(experience_idx, current_examples))

        replay_pool = [idx for exp in self.experiences[:experience_idx] for idx in exp]
        replay_examples = max(target_budget - current_examples, 0)
        if replay_examples > 0:
            source_pool = replay_pool if replay_pool else active_indices
            if source_pool:
                rng = np.random.default_rng(self.seed + client_id * 1009 + server_round)
                if self.sample_with_replacement or replay_examples > len(source_pool):
                    sampled = rng.choice(source_pool, size=replay_examples, replace=True).tolist()
                else:
                    sampled = rng.choice(source_pool, size=replay_examples, replace=False).tolist()
                active_indices.extend(sampled)
            else:
                replay_examples = 0

        unique_examples = len(set(active_indices))
        return ContinualRoundSelection(
            indices=active_indices,
            current_examples=current_examples,
            replay_examples=replay_examples,
            unique_examples=unique_examples,
        )


def build_continual_schedule(
    indices: List[int],
    num_experiences: int,
    seed: int,
    client_id: int,
    replay_ratio: float,
    budget_mode: str,
) -> Optional[ContinualSchedule]:
    if not indices:
        return None

    shuffled = np.array(indices, dtype=np.int64)
    rng = np.random.default_rng(seed + client_id * 97)
    rng.shuffle(shuffled)

    effective_experiences = max(1, min(num_experiences, len(shuffled)))
    chunks = [chunk.tolist() for chunk in np.array_split(shuffled, effective_experiences) if len(chunk) > 0]
    return ContinualSchedule(
        experiences=chunks,
        replay_ratio=replay_ratio,
        seed=seed,
        budget_mode=budget_mode,
    )

class FlowerClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        trainset,
        train_indices,
        valloader,
        device,
        epochs,
        lr,
        batch_size,
        client_id,
        forward_flops_per_example=0.0,
        back_model=None,
        continual_schedule: Optional[ContinualSchedule] = None,
    ):
        self.model = model
        self.back_model = back_model
        self.trainset = trainset
        self.train_indices = list(train_indices)
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.client_id = int(client_id)
        self.forward_flops_per_example = float(forward_flops_per_example)
        self.continual_schedule = continual_schedule

    def _state_keys(self):
        front_keys = list(self.model.state_dict().keys())
        back_keys = list(self.back_model.state_dict().keys()) if self.back_model is not None else []
        return front_keys, back_keys

    def _select_round_indices(self, server_round: int) -> ContinualRoundSelection:
        active_indices = list(self.train_indices)
        selection = ContinualRoundSelection(
            indices=active_indices,
            current_examples=len(active_indices),
            replay_examples=0,
            unique_examples=len(set(active_indices)),
        )
        if self.continual_schedule is not None:
            selection = self.continual_schedule.indices_for_round(server_round, self.client_id)
        return selection

    def _make_trainloader(self, round_selection: ContinualRoundSelection) -> DataLoader:
        subset = Subset(self.trainset, round_selection.indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    def get_parameters(self, config):
        params = [val.detach().cpu().numpy() for _, val in self.model.state_dict().items()]
        if self.back_model is not None:
            params.extend([val.detach().cpu().numpy() for _, val in self.back_model.state_dict().items()])
        return params

    def set_parameters(self, parameters):
        front_keys, back_keys = self._state_keys()
        front_count = len(front_keys)

        front_state = OrderedDict(
            {k: torch.tensor(v) for k, v in zip(front_keys, parameters[:front_count])}
        )
        self.model.load_state_dict(front_state, strict=True)

        if self.back_model is not None:
            back_state = OrderedDict(
                {k: torch.tensor(v) for k, v in zip(back_keys, parameters[front_count:])}
            )
            self.back_model.load_state_dict(back_state, strict=True)

    def fit(self, parameters, config):
        fit_wall_start = time.perf_counter()
        set_parameters_start = time.perf_counter()
        self.set_parameters(parameters)
        receive_time_sec = time.perf_counter() - set_parameters_start
        server_round = int(config.get("server_round", 1))
        round_selection = self._select_round_indices(server_round)
        trainloader = self._make_trainloader(round_selection)
        active_examples = len(trainloader.dataset)
        unique_examples = int(round_selection.unique_examples)
        replay_examples = int(round_selection.replay_examples)
        current_examples = int(round_selection.current_examples)

        if active_examples == 0:
            send_parameters_start = time.perf_counter()
            outgoing_parameters = self.get_parameters(config={})
            send_time_sec = time.perf_counter() - send_parameters_start
            fit_wall_time_sec = time.perf_counter() - fit_wall_start
            training_time_sec = 0.0
            communication_time_sec = receive_time_sec + send_time_sec
            fit_overhead_time_sec = max(
                fit_wall_time_sec - training_time_sec - communication_time_sec,
                0.0,
            )
            return outgoing_parameters, 0, {
                "loss": 0.0,
                "active_examples": 0,
                "unique_train_examples": 0,
                "current_examples": 0,
                "replay_examples": 0,
                "training_time_sec": float(training_time_sec),
                "communication_time_sec": float(communication_time_sec),
                "fit_wall_time_sec": float(fit_wall_time_sec),
                "fit_overhead_time_sec": float(fit_overhead_time_sec),
            }

        self.model.train()
        if self.back_model is not None:
            self.back_model.train()

        optimizer_front = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        optimizer_back = (
            optim.SGD(self.back_model.parameters(), lr=self.lr, momentum=0.9)
            if self.back_model is not None
            else None
        )
        criterion = nn.CrossEntropyLoss()

        epoch_losses = []
        training_start = time.perf_counter()
        for _ in range(self.epochs):
            batch_losses = []
            for data, target in trainloader:
                data, target = data.to(self.device), target.to(self.device).long()

                if self.back_model is None:
                    optimizer_front.zero_grad()
                    output = self.model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer_front.step()
                    batch_losses.append(loss.item())
                    continue

                optimizer_front.zero_grad()
                optimizer_back.zero_grad()

                smashed_data = self.model(data)
                detached_smashed = smashed_data.detach().clone().requires_grad_(True)
                output = self.back_model(detached_smashed)
                loss = criterion(output, target)
                loss.backward()
                optimizer_back.step()

                smashed_grad = detached_smashed.grad.clone()
                optimizer_front.zero_grad()
                smashed_data.backward(smashed_grad)
                optimizer_front.step()
                batch_losses.append(loss.item())

            if batch_losses:
                epoch_losses.append(sum(batch_losses) / len(batch_losses))
        training_time_sec = time.perf_counter() - training_start

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        total_training_examples = active_examples * self.epochs
        fit_flops = self.forward_flops_per_example * TRAIN_FLOP_MULTIPLIER * total_training_examples
        send_parameters_start = time.perf_counter()
        outgoing_parameters = self.get_parameters(config={})
        send_time_sec = time.perf_counter() - send_parameters_start
        fit_wall_time_sec = time.perf_counter() - fit_wall_start
        # Application-level communication proxy: parameter ingest/export handled inside the client fit call.
        communication_time_sec = receive_time_sec + send_time_sec
        fit_overhead_time_sec = max(
            fit_wall_time_sec - training_time_sec - communication_time_sec,
            0.0,
        )

        return outgoing_parameters, active_examples, {
            "loss": float(avg_loss),
            "active_examples": int(active_examples),
            "unique_train_examples": unique_examples,
            "current_examples": current_examples,
            "replay_examples": replay_examples,
            "fit_flops": float(fit_flops),
            "training_time_sec": float(training_time_sec),
            "communication_time_sec": float(communication_time_sec),
            "fit_wall_time_sec": float(fit_wall_time_sec),
            "fit_overhead_time_sec": float(fit_overhead_time_sec),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        if self.back_model is not None:
            self.back_model.eval()

        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        loss = 0.0
        with torch.no_grad():
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device).long()
                output = self.model(data)
                if self.back_model is not None:
                    output = self.back_model(output)

                loss += criterion(output, target).item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        acc = correct / total if total > 0 else 0
        eval_flops = self.forward_flops_per_example * total
        return float(loss) / len(self.valloader), len(self.valloader.dataset), {
            "accuracy": float(acc),
            "eval_flops": float(eval_flops),
        }


class SplitFedClient(fl.client.NumPyClient):
    def __init__(
        self,
        model,
        trainset,
        train_indices,
        valloader,
        device,
        epochs,
        lr,
        batch_size,
        client_id,
        server_actor_name,
        sfl_variant: str,
        server_actor_pool_names=None,
        forward_flops_per_example=0.0,
    ):
        self.model = model
        self.trainset = trainset
        self.train_indices = list(train_indices)
        self.valloader = valloader
        self.device = device
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.client_id = int(client_id)
        self.server_actor_name = str(server_actor_name)
        self.server_actor_pool_names = list(server_actor_pool_names) if server_actor_pool_names is not None else None
        self.sfl_variant = str(sfl_variant)
        self.forward_flops_per_example = float(forward_flops_per_example)
        self._server_actor_cache = {}

    def get_parameters(self, config):
        return model_state_to_numpy(self.model)

    def set_parameters(self, parameters):
        load_numpy_state(self.model, parameters)

    def _make_trainloader(self) -> DataLoader:
        subset = Subset(self.trainset, self.train_indices)
        return DataLoader(subset, batch_size=self.batch_size, shuffle=True)

    def _get_named_actor(self, actor_name: str):
        if actor_name not in self._server_actor_cache:
            self._server_actor_cache[actor_name] = ray.get_actor(actor_name)
        return self._server_actor_cache[actor_name]

    def _call_server_train(
        self,
        server_round: int,
        fit_config: Dict[str, Scalar],
        smashed_data: np.ndarray,
        labels: np.ndarray,
    ) -> Tuple[Dict[str, Union[str, float, np.ndarray]], float]:
        if self.sfl_variant == "SFLV1":
            slot = int(fit_config.get("sfl_server_slot", 0))
            actor = self._get_named_actor(self.server_actor_pool_names[slot])
            start_time = time.perf_counter()
            result = ray.get(actor.train_batch.remote(smashed_data, labels))
            return result, time.perf_counter() - start_time

        total_elapsed = 0.0
        while True:
            start_time = time.perf_counter()
            result = ray.get(
                self._get_named_actor(self.server_actor_name).train_batch.remote(
                    int(server_round),
                    int(fit_config.get("sfl_v2_turn", 0)),
                    smashed_data,
                    labels,
                )
            )
            total_elapsed += time.perf_counter() - start_time
            status = str(result.get("status", ""))
            if status == "ok":
                return result, total_elapsed
            if status not in {"wait", "stale"}:
                raise RuntimeError(f"Unexpected SFL-v2 server status: {status}")
            time.sleep(0.01)

    def _finish_server_round(self, server_round: int) -> float:
        if self.sfl_variant != "SFLV2":
            return 0.0

        actor = self._get_named_actor(self.server_actor_name)
        total_elapsed = 0.0
        client_turn = int(getattr(self, "_active_sfl_v2_turn", 0))
        while True:
            start_time = time.perf_counter()
            result = ray.get(actor.finish_client.remote(int(server_round), client_turn))
            total_elapsed += time.perf_counter() - start_time
            status = str(result.get("status", ""))
            if status == "ok" or status == "done":
                return total_elapsed
            if status not in {"wait", "stale"}:
                raise RuntimeError(f"Unexpected SFL-v2 finish status: {status}")
            time.sleep(0.01)

    def _call_server_eval(self, smashed_data: np.ndarray, labels: np.ndarray) -> Tuple[Dict[str, float], float]:
        if self.sfl_variant == "SFLV2":
            actor_name = self.server_actor_name
        else:
            if not self.server_actor_pool_names:
                raise RuntimeError("SFL-v1 server actor pool is empty.")
            eval_slot = int(self.client_id) % len(self.server_actor_pool_names)
            actor_name = self.server_actor_pool_names[eval_slot]
        actor = self._get_named_actor(actor_name)
        start_time = time.perf_counter()
        result = ray.get(actor.evaluate_batch.remote(smashed_data, labels))
        return result, time.perf_counter() - start_time

    def fit(self, parameters, config):
        fit_wall_start = time.perf_counter()
        set_parameters_start = time.perf_counter()
        self.set_parameters(parameters)
        receive_time_sec = time.perf_counter() - set_parameters_start
        server_round = int(config.get("server_round", 1))
        self._active_sfl_v2_turn = int(config.get("sfl_v2_turn", 0))
        trainloader = self._make_trainloader()
        active_examples = len(trainloader.dataset)

        if active_examples == 0:
            send_parameters_start = time.perf_counter()
            outgoing_parameters = self.get_parameters(config={})
            send_time_sec = time.perf_counter() - send_parameters_start
            fit_wall_time_sec = time.perf_counter() - fit_wall_start
            communication_time_sec = receive_time_sec + send_time_sec
            return outgoing_parameters, 0, {
                "loss": 0.0,
                "active_examples": 0,
                "unique_train_examples": 0,
                "current_examples": 0,
                "replay_examples": 0,
                "training_time_sec": 0.0,
                "communication_time_sec": float(communication_time_sec),
                "fit_wall_time_sec": float(fit_wall_time_sec),
                "fit_overhead_time_sec": float(max(fit_wall_time_sec - communication_time_sec, 0.0)),
            }

        self.model.train()
        optimizer_front = optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        epoch_losses = []
        local_compute_time_sec = 0.0
        server_compute_time_sec = 0.0
        communication_time_sec = receive_time_sec

        for _ in range(self.epochs):
            batch_losses = []
            for data, target in trainloader:
                data = data.to(self.device)
                target = target.to(self.device).long()

                optimizer_front.zero_grad()

                forward_start = time.perf_counter()
                smashed_data = self.model(data)
                sync_device(self.device)
                local_compute_time_sec += time.perf_counter() - forward_start

                result, remote_elapsed = self._call_server_train(
                    server_round=server_round,
                    fit_config=config,
                    smashed_data=smashed_data.detach().cpu().numpy(),
                    labels=target.detach().cpu().numpy(),
                )
                server_time = float(result.get("server_compute_time_sec", 0.0))
                server_compute_time_sec += server_time
                communication_time_sec += max(remote_elapsed - server_time, 0.0)

                backward_start = time.perf_counter()
                smashed_grad = torch.tensor(
                    result["smashed_grad"],
                    dtype=smashed_data.dtype,
                    device=self.device,
                )
                smashed_data.backward(smashed_grad)
                optimizer_front.step()
                sync_device(self.device)
                local_compute_time_sec += time.perf_counter() - backward_start
                batch_losses.append(float(result.get("loss", 0.0)))

            if batch_losses:
                epoch_losses.append(sum(batch_losses) / len(batch_losses))

        communication_time_sec += self._finish_server_round(server_round)

        avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0.0
        training_time_sec = local_compute_time_sec + server_compute_time_sec
        total_training_examples = active_examples * self.epochs
        fit_flops = self.forward_flops_per_example * TRAIN_FLOP_MULTIPLIER * total_training_examples

        send_parameters_start = time.perf_counter()
        outgoing_parameters = self.get_parameters(config={})
        send_time_sec = time.perf_counter() - send_parameters_start
        communication_time_sec += send_time_sec
        fit_wall_time_sec = time.perf_counter() - fit_wall_start
        fit_overhead_time_sec = max(
            fit_wall_time_sec - training_time_sec - communication_time_sec,
            0.0,
        )

        return outgoing_parameters, active_examples, {
            "loss": float(avg_loss),
            "active_examples": int(active_examples),
            "unique_train_examples": int(active_examples),
            "current_examples": int(active_examples),
            "replay_examples": 0,
            "fit_flops": float(fit_flops),
            "training_time_sec": float(training_time_sec),
            "communication_time_sec": float(communication_time_sec),
            "fit_wall_time_sec": float(fit_wall_time_sec),
            "fit_overhead_time_sec": float(fit_overhead_time_sec),
        }

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total = 0
        correct = 0
        loss_sum = 0.0
        with torch.no_grad():
            for data, target in self.valloader:
                data = data.to(self.device)
                target = target.to(self.device).long()
                smashed_data = self.model(data)
                result, _ = self._call_server_eval(
                    smashed_data=smashed_data.detach().cpu().numpy(),
                    labels=target.detach().cpu().numpy(),
                )
                total += int(result.get("total", 0.0))
                correct += int(result.get("correct", 0.0))
                loss_sum += float(result.get("loss_sum", 0.0))

        accuracy = correct / total if total > 0 else 0.0
        eval_flops = self.forward_flops_per_example * total
        average_loss = loss_sum / total if total > 0 else 0.0
        return float(average_loss), len(self.valloader.dataset), {
            "accuracy": float(accuracy),
            "eval_flops": float(eval_flops),
        }

# --- 3. Performance Tracker ---

class PerformanceTracker:
    def __init__(self):
        self.fit_start_time = 0
        self.last_fit_aggregate_time = 0
        self.durations = []
        self.tflops = []
        self._last_fit_summary = {
            "fit_flops": 0.0,
            "train_examples": 0.0,
            "unique_train_examples": 0.0,
            "current_examples": 0.0,
            "replay_examples": 0.0,
            "fit_participants": 0.0,
            "training_time_sum_sec": 0.0,
            "communication_time_sum_sec": 0.0,
            "fit_wall_time_sum_sec": 0.0,
            "fit_overhead_time_sum_sec": 0.0,
        }

    def start_fit(self, server_round: int):
        if self.fit_start_time <= 0:
            self.fit_start_time = time.time()
            self.last_fit_aggregate_time = 0
            self._last_fit_summary = {
                "fit_flops": 0.0,
                "train_examples": 0.0,
                "unique_train_examples": 0.0,
                "current_examples": 0.0,
                "replay_examples": 0.0,
                "fit_participants": 0.0,
                "training_time_sum_sec": 0.0,
                "communication_time_sum_sec": 0.0,
                "fit_wall_time_sum_sec": 0.0,
                "fit_overhead_time_sum_sec": 0.0,
            }
        return {"server_round": int(server_round)}

    def aggregate_fit_metrics(self, metrics: List[Tuple[int, Dict]]):
        increment = {
            "fit_flops": sum(float(m.get("fit_flops", 0.0)) for _, m in metrics),
            "train_examples": sum(float(m.get("active_examples", 0.0)) for _, m in metrics),
            "unique_train_examples": sum(float(m.get("unique_train_examples", 0.0)) for _, m in metrics),
            "current_examples": sum(float(m.get("current_examples", 0.0)) for _, m in metrics),
            "replay_examples": sum(float(m.get("replay_examples", 0.0)) for _, m in metrics),
            "fit_participants": float(len(metrics)),
            "training_time_sum_sec": sum(float(m.get("training_time_sec", 0.0)) for _, m in metrics),
            "communication_time_sum_sec": sum(float(m.get("communication_time_sec", 0.0)) for _, m in metrics),
            "fit_wall_time_sum_sec": sum(float(m.get("fit_wall_time_sec", 0.0)) for _, m in metrics),
            "fit_overhead_time_sum_sec": sum(float(m.get("fit_overhead_time_sec", 0.0)) for _, m in metrics),
        }
        for key, value in increment.items():
            self._last_fit_summary[key] += value
        self.last_fit_aggregate_time = time.time()
        return {key: float(value) for key, value in self._last_fit_summary.items()}

    def stop_evaluate(self, metrics: List[Tuple[int, Dict]]):
        # This is where we finalize the round duration
        duration = 0.0
        if self.fit_start_time > 0:
            duration = time.time() - self.fit_start_time
            self.durations.append(duration)
        
        if not metrics:
            self.tflops.append(0.0)
            return {"accuracy": 0.0, "tflops": 0.0}
            
        accs = [num_examples * m["accuracy"] for num_examples, m in metrics]
        examples = [num_examples for num_examples, _ in metrics]
        total_eval_flops = sum(float(m.get("eval_flops", 0.0)) for _, m in metrics)
        total_round_flops = self._last_fit_summary["fit_flops"] + total_eval_flops
        round_tflops = (
            total_round_flops / duration / 1e12
            if duration > 0.0
            else 0.0
        )
        self.tflops.append(round_tflops)
        fit_summary = dict(self._last_fit_summary)
        fit_participants = max(int(round(fit_summary.get("fit_participants", 0.0))), 0)
        avg_training_time_sec = (
            fit_summary["training_time_sum_sec"] / fit_participants
            if fit_participants > 0
            else 0.0
        )
        avg_communication_time_sec = (
            fit_summary["communication_time_sum_sec"] / fit_participants
            if fit_participants > 0
            else 0.0
        )
        avg_fit_wall_time_sec = (
            fit_summary["fit_wall_time_sum_sec"] / fit_participants
            if fit_participants > 0
            else 0.0
        )
        avg_fit_overhead_time_sec = (
            fit_summary["fit_overhead_time_sum_sec"] / fit_participants
            if fit_participants > 0
            else 0.0
        )
        fit_phase_duration_sec = 0.0
        if self.fit_start_time > 0 and self.last_fit_aggregate_time > 0:
            fit_phase_duration_sec = max(self.last_fit_aggregate_time - self.fit_start_time, 0.0)
        self.fit_start_time = 0
        self.last_fit_aggregate_time = 0
        self._last_fit_summary = {
            "fit_flops": 0.0,
            "train_examples": 0.0,
            "unique_train_examples": 0.0,
            "current_examples": 0.0,
            "replay_examples": 0.0,
            "fit_participants": 0.0,
            "training_time_sum_sec": 0.0,
            "communication_time_sum_sec": 0.0,
            "fit_wall_time_sum_sec": 0.0,
            "fit_overhead_time_sum_sec": 0.0,
        }
        return {
            "accuracy": sum(accs) / sum(examples),
            "tflops": float(round_tflops),
            "fit_flops": float(fit_summary["fit_flops"]),
            "eval_flops": float(total_eval_flops),
            "train_examples": float(fit_summary["train_examples"]),
            "unique_train_examples": float(fit_summary["unique_train_examples"]),
            "current_examples": float(fit_summary["current_examples"]),
            "replay_examples": float(fit_summary["replay_examples"]),
            "eval_examples": float(sum(examples)),
            "avg_training_time_sec": float(avg_training_time_sec),
            "avg_communication_time_sec": float(avg_communication_time_sec),
            "avg_fit_time_sec": float(avg_fit_wall_time_sec),
            "avg_fit_overhead_time_sec": float(avg_fit_overhead_time_sec),
            "fit_phase_duration_sec": float(fit_phase_duration_sec),
        }

# --- 4. Utilities ---


def chunk_list(values: List[float], group_size: int) -> List[List[float]]:
    if group_size <= 1:
        return [list(values)]
    return [values[i:i + group_size] for i in range(0, len(values), group_size)]


def collapse_metric_series(
    history: fl.server.history.History,
    metric_name: str,
    group_size: int,
    reducer: str = "last",
) -> List[float]:
    if metric_name not in history.metrics_distributed:
        return []
    values = [float(val) for _, val in history.metrics_distributed[metric_name]]
    if group_size <= 1:
        return values

    collapsed = []
    for group in chunk_list(values, group_size):
        if not group:
            continue
        if reducer == "sum":
            collapsed.append(float(sum(group)))
        elif reducer == "mean":
            collapsed.append(float(sum(group) / len(group)))
        else:
            collapsed.append(float(group[-1]))
    return collapsed


def collapse_sequential_round_results(
    history: fl.server.history.History,
    durations: List[float],
    group_size: int,
) -> Dict[str, List[float]]:
    accuracies = collapse_metric_series(history, "accuracy", group_size, reducer="last")
    fit_flops = collapse_metric_series(history, "fit_flops", group_size, reducer="sum")
    eval_flops = collapse_metric_series(history, "eval_flops", group_size, reducer="sum")
    durations_collapsed = [
        float(sum(group))
        for group in chunk_list(durations, group_size)
        if group
    ]
    tflops = []
    for idx, duration in enumerate(durations_collapsed):
        total_flops = 0.0
        if idx < len(fit_flops):
            total_flops += fit_flops[idx]
        if idx < len(eval_flops):
            total_flops += eval_flops[idx]
        tflops.append(total_flops / duration / 1e12 if duration > 0.0 else 0.0)

    return {
        "accuracy": accuracies,
        "durations": durations_collapsed,
        "tflops": tflops,
        "train_examples": collapse_metric_series(history, "train_examples", group_size, reducer="sum"),
        "unique_train_examples": collapse_metric_series(history, "unique_train_examples", group_size, reducer="sum"),
        "current_examples": collapse_metric_series(history, "current_examples", group_size, reducer="sum"),
        "replay_examples": collapse_metric_series(history, "replay_examples", group_size, reducer="sum"),
        "eval_examples": collapse_metric_series(history, "eval_examples", group_size, reducer="sum"),
        "avg_training_time_sec": collapse_metric_series(history, "avg_training_time_sec", group_size, reducer="mean"),
        "avg_communication_time_sec": collapse_metric_series(history, "avg_communication_time_sec", group_size, reducer="mean"),
        "avg_fit_time_sec": collapse_metric_series(history, "avg_fit_time_sec", group_size, reducer="mean"),
        "avg_fit_overhead_time_sec": collapse_metric_series(history, "avg_fit_overhead_time_sec", group_size, reducer="mean"),
        "fit_phase_duration_sec": collapse_metric_series(history, "fit_phase_duration_sec", group_size, reducer="sum"),
        "fit_flops": fit_flops,
        "eval_flops": eval_flops,
    }

def print_experiment_summary(
    args,
    device,
    client_indices,
    targets,
    selected_baselines,
    num_classes,
    learning_type,
    effective_num_clients,
    participating_clients_per_round,
):
    print("\n" + "="*50)
    print("      EXPERIMENT CONFIGURATION SUMMARY")
    print("="*50)
    print(f"Dataset:       {args.dataset.upper()}")
    print(f"Process Unit:  {device.upper()}")
    print(f"Rounds:        {args.rounds}")
    print(f"Local Epochs:  {args.epochs}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Num Clients:   {effective_num_clients}")
    print(f"Clients/Round: {participating_clients_per_round}")
    print(f"Data Distr:    Alpha={args.data_distr} ({'IID' if args.data_distr >= 1.0 else 'Non-IID'})")
    print(f"Baselines:     {', '.join([STRATEGY_DISPLAY_NAMES.get(b, b) for b in selected_baselines])}")
    print(f"Model:         {args.model}")
    print(f"Learning Type: {LEARNING_TYPE_DISPLAY_NAMES[learning_type]}")
    print(f"Compare Mode:  {args.comparison_profile}")
    if learning_type in {'CFL', 'CFSL'}:
        print(
            f"Continual CFG: steps={args.continual_steps}, "
            f"replay_ratio={args.continual_replay_ratio}, "
            f"budget_mode={args.continual_budget_mode}"
        )
    print("-" * 50)
    
    print("Data Distribution per Client (Label Counts):")
    # Truncate label header if too many classes
    max_labels_to_show = 15
    if num_classes > max_labels_to_show:
        labels_to_show = list(range(max_labels_to_show))
        label_header = " | ".join([f"L{i}" for i in labels_to_show]) + " | ..."
    else:
        label_header = " | ".join([f"L{i}" for i in range(num_classes)])
    
    header = f"Client    | {label_header} | Total"
    print(header)
    print("-" * len(header))
    
    for client_idx in range(len(client_indices)):
        client_targets = targets[client_indices[client_idx]]
        counts = []
        limit = min(num_classes, max_labels_to_show)
        for label in range(limit):
            count = np.sum(client_targets == label)
            counts.append(f"{count:3}")
        
        row = f"Client {client_idx:2} | " + " | ".join(counts)
        if num_classes > max_labels_to_show:
            row += " | ..."
        row += f" | {len(client_targets):4}"
        print(row)
    print("="*50 + "\n")

# --- 5. Strategy Naming Mapping ---

STRATEGY_DISPLAY_NAMES = {
    'centralized': 'Centralized',
    'fedavg': 'FedAvg',
    #'fedavgm': 'FedAvgM',
    #'fedmedian': 'FedMedian',
    #'fedadam': 'FedAdam',
    #'fedprox': 'FedProx',
    'fedgsw': 'FedGSW',
    'fedlaa': 'FedLAA',
    'fedama': 'FedAMA',
    'fedde': 'FedDE',
    'fedsbfl': 'FedSBFL',
    'splitseq': 'SplitSeq',
}



# --- 6. Main Simulation ---

def main():
    parser = argparse.ArgumentParser(description='Flower Baseline Simulator')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'stl10', 'mnist', 'oxfordpet', 'adult', 'speechcommands'])
    parser.add_argument('--model', type=str, nargs='+', default=['cnn'], 
                        help='Model(s) to use. Can specify multiple: --model cnn resnet',
                        choices=['cnn', 'squeezenet', 'shufflenet', 'resnet', 'mlp', 'm5'])
    parser.add_argument('--num_clients', type=int, default=4)
    parser.add_argument(
        '--clients-per-round',
        type=int,
        default=None,
        help='Number of participating clients per round (default: all instantiated clients)',
    )
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=1, help='Local epochs (auto-tuned if None)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size (auto-tuned if None)')
    parser.add_argument('--baseline', type=str, nargs='+', default=['all'], 
                        help='Aggregation strategy or "all". Available: ' + ', '.join(STRATEGY_DISPLAY_NAMES.keys()))
    parser.add_argument('--lr', type=float, default=None, help='Learning rate (auto-tuned if None)')
    parser.add_argument('--data-distr', type=float, default=1.0, help='Data distribution (1.0 for IID, < 1.0 for Dirichlet non-IID)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument(
        '--learning-type',
        type=str,
        default='FL',
        help='Learning type: CL, FL, SL, SFLV1, SFLV2, CFL, or CFSL',
    )
    parser.add_argument(
        '--continual-steps',
        type=int,
        default=None,
        help='Number of continual experiences per client (defaults to rounds)',
    )
    parser.add_argument(
        '--continual-replay-ratio',
        type=float,
        default=None,
        help='Replay ratio used in continual modes',
    )
    parser.add_argument(
        '--comparison-profile',
        type=str,
        default='fair',
        choices=['fair', 'legacy'],
        help='Comparison profile: fair uses logical SL rounds and budget-matched continual replay',
    )
    
    args = parser.parse_args()
    args.model = [m.lower() for m in args.model]
    args.learning_type = normalize_learning_type(args.learning_type)
    if args.continual_steps is None:
        args.continual_steps = min(args.rounds, 4) if args.comparison_profile == 'fair' else args.rounds
    args.continual_steps = max(1, int(args.continual_steps))
    if args.continual_replay_ratio is None:
        args.continual_replay_ratio = 1.0 if args.comparison_profile == 'fair' else 0.25
    args.continual_replay_ratio = max(0.0, float(args.continual_replay_ratio))
    args.continual_budget_mode = 'partition' if args.comparison_profile == 'fair' else 'experience'

    # Set random seed
    random_seed = args.seed
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
    
    hparams = DATASET_HYPERPARAMS.get(args.dataset.lower(), {'lr': 0.01, 'batch_size': 32, 'epochs': 1})
    if args.lr is None: args.lr = hparams['lr']
    if args.batch_size is None: args.batch_size = hparams['batch_size']
    if args.epochs is None: args.epochs = hparams['epochs']
    
    args.vis_dir = 'results'
    os.makedirs(args.vis_dir, exist_ok=True)
    os.makedirs('csv', exist_ok=True)
    
    if torch.cuda.is_available():
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    else:
        device = 'cpu'

    is_tabular = args.dataset in ['adult']
    is_audio = args.dataset in ['speechcommands']
    uses_splitfed = args.learning_type in {'SFLV1', 'SFLV2'}
    uses_local_split = args.learning_type in {'SL', 'CFSL'}
    uses_split = uses_local_split or uses_splitfed
    uses_continual = args.learning_type in {'CFL', 'CFSL'}
    is_centralized = args.learning_type == 'CL'
    is_split_sequential = args.learning_type == 'SL'
    is_splitfed_v1 = args.learning_type == 'SFLV1'
    is_splitfed_v2 = args.learning_type == 'SFLV2'

    models_to_run = [m.lower() for m in args.model]

    if is_tabular:
        img_size = None
        in_channels = None
        models_to_run = ['mlp']
    elif is_audio:
        img_size = 8000
        in_channels = 1
        models_to_run = ['m5']
    else:
        if args.dataset == 'mnist':
            img_size = 28
        elif args.dataset == 'oxfordpet':
            img_size = 224
        elif args.dataset == 'cifar10':
            img_size = 32
        else:
            img_size = 128

        in_channels = 1 if args.dataset == 'mnist' else 3

        for m in models_to_run:
            if m in ['mlp', 'm5']:
                raise ValueError(f"Model {m} not supported for image dataset {args.dataset}")

    train_set, test_set, targets, num_classes = get_dataset(
        args.dataset,
        img_size if not (is_tabular or is_audio) else None,
    )
    input_dim = train_set.X.shape[1] if is_tabular else None

    effective_num_clients = 1 if is_centralized else args.num_clients
    if args.clients_per_round is None:
        args.clients_per_round = effective_num_clients
    args.clients_per_round = max(1, min(int(args.clients_per_round), effective_num_clients))
    if is_centralized:
        client_indices = [list(range(len(train_set)))]
    else:
        client_indices = partition_data(train_set, targets, effective_num_clients, args.data_distr, num_classes)

    federated_baselines = [
        baseline for baseline in STRATEGY_DISPLAY_NAMES.keys()
        if baseline not in {'centralized', 'splitseq'}
    ]
    requested_baselines = [b.lower() for b in args.baseline]
    if is_centralized:
        selected_baselines = ['centralized']
    elif is_split_sequential:
        selected_baselines = ['splitseq']
    elif uses_splitfed:
        selected_baselines = ['fedavg']
    elif 'all' in requested_baselines:
        selected_baselines = federated_baselines
    else:
        invalid_baselines = [b for b in requested_baselines if b not in federated_baselines]
        if invalid_baselines:
            raise ValueError(
                f"Unknown baselines: {', '.join(invalid_baselines)}. "
                f"Available: {', '.join(federated_baselines)}"
            )
        selected_baselines = requested_baselines

    if uses_splitfed and requested_baselines not in (['all'], ['fedavg']):
        print("  - SplitFed follows the paper with FedAvg aggregation; overriding requested baseline(s) to FedAvg.")

    def file_mode_name(mode: str) -> str:
        if mode in {'centralized', 'splitseq'}:
            return mode
        return f"{mode}-{LEARNING_TYPE_FILE_TAGS[args.learning_type]}"

    baseline_name_for_file = "-".join(file_mode_name(mode) for mode in selected_baselines)

    for model_name in models_to_run:
        args.model = model_name

        if is_tabular:
            input_dim = train_set.X.shape[1]
        else:
            input_dim = None

        def build_full_model():
            if is_tabular:
                return get_model(model_name, num_classes=num_classes, input_dim=input_dim)
            if is_audio:
                return get_model(model_name, num_classes=num_classes, in_channels=in_channels)
            return get_model(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)

        def build_split_model_pair():
            if is_tabular:
                return get_split_models(model_name, num_classes=num_classes, input_dim=input_dim)
            if is_audio:
                return get_split_models(model_name, num_classes=num_classes, in_channels=in_channels)
            return get_split_models(model_name, num_classes=num_classes, input_size=img_size, in_channels=in_channels)

        continual_schedules = {}
        if uses_continual:
            for client_id, indices in enumerate(client_indices):
                continual_schedules[client_id] = build_continual_schedule(
                    indices=indices,
                    num_experiences=args.continual_steps,
                    seed=args.seed,
                    client_id=client_id,
                    replay_ratio=args.continual_replay_ratio,
                    budget_mode=args.continual_budget_mode,
                )

        forward_flops_per_example = estimate_model_flops(
            model_name=model_name,
            num_classes=num_classes,
            is_split=uses_split,
            input_dim=input_dim,
            in_channels=in_channels,
            img_size=img_size,
        )
        print(f"  - Estimated forward FLOPs/example: {forward_flops_per_example:.0f}")

        splitfed_server_actor_names: List[str] = []
        splitfed_shared_actor_name = ""
        splitfed_initial_server_parameters: List[np.ndarray] = []
        splitfed_actor_kwargs: Dict[str, Union[str, int, float, None]] = {}
        if uses_splitfed:
            _, initial_back_model = build_split_model_pair()
            splitfed_initial_server_parameters = model_state_to_numpy(initial_back_model)
            splitfed_actor_kwargs = {
                "model_name": model_name,
                "num_classes": num_classes,
                "lr": args.lr,
                "device": device,
                "input_dim": input_dim,
                "in_channels": in_channels,
                "img_size": img_size,
            }
            actor_base_name = (
                f"splitfed_{args.learning_type.lower()}_{model_name}_{args.dataset}_"
                f"{os.getpid()}_{int(time.time() * 1_000_000)}"
            )
            if is_splitfed_v1:
                splitfed_server_actor_names = [
                    f"{actor_base_name}_slot_{slot}"
                    for slot in range(args.clients_per_round)
                ]
            else:
                splitfed_shared_actor_name = f"{actor_base_name}_shared"

        if uses_split:
            warmup_front, warmup_back = build_split_model_pair()
            warmup_model = warmup_front.to(device)
            warmup_back = warmup_back.to(device)
        else:
            warmup_model = build_full_model().to(device)
            warmup_back = None

        def client_fn(context: Context) -> fl.client.Client:
            cid = int(context.node_config["partition-id"])
            if uses_split:
                model, back_model = build_split_model_pair()
                model = model.to(device)
                if uses_local_split:
                    back_model = back_model.to(device)
                else:
                    back_model = None
            else:
                model = build_full_model().to(device)
                back_model = None

            model = model.to(device)
            valloader = DataLoader(test_set, batch_size=args.batch_size)
            if uses_splitfed:
                return SplitFedClient(
                    model=model,
                    trainset=train_set,
                    train_indices=client_indices[cid],
                    valloader=valloader,
                    device=device,
                    epochs=args.epochs,
                    lr=args.lr,
                    batch_size=args.batch_size,
                    client_id=cid,
                    server_actor_name=splitfed_shared_actor_name if is_splitfed_v2 else splitfed_server_actor_names[0],
                    server_actor_pool_names=splitfed_server_actor_names if is_splitfed_v1 else None,
                    sfl_variant=args.learning_type,
                    forward_flops_per_example=forward_flops_per_example,
                ).to_client()
            return FlowerClient(
                model=model,
                trainset=train_set,
                train_indices=client_indices[cid],
                valloader=valloader,
                device=device,
                epochs=args.epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                client_id=cid,
                forward_flops_per_example=forward_flops_per_example,
                back_model=back_model,
                continual_schedule=continual_schedules.get(cid),
            ).to_client()

        dummy_input = torch.randn(1, input_dim).to(device) if is_tabular else (
            torch.randn(1, in_channels, img_size).to(device) if is_audio else 
            torch.randn(1, in_channels, img_size, img_size).to(device)
        )

        print("  - Warming up hardware (first-run initialization)...")
        warmup_model.train()
        try:
            output = warmup_model(dummy_input)
            if warmup_back is not None:
                output = warmup_back(output)
            loss = output.sum()
            loss.backward()
            if device == 'cuda':
                torch.cuda.synchronize()
            elif device == 'mps':
                torch.mps.synchronize()
        except Exception as e:
            print(f"    Warning: Warmup failed (not critical): {e}")
        print("  - Warmup complete.")

        print_experiment_summary(
            args,
            device,
            client_indices,
            targets,
            selected_baselines,
            num_classes,
            args.learning_type,
            effective_num_clients,
            args.clients_per_round,
        )

        all_metric_results = {}
        all_time_results = {}
        all_tflops_results = {}
        all_avg_training_time_results = {}
        all_avg_communication_time_results = {}
        all_fit_phase_duration_results = {}
        for mode in selected_baselines:
            history_acc = []
            durations_fixed = []
            history_tflops = []
            history_avg_training_time = []
            history_avg_communication_time = []
            history_avg_fit_time = []
            history_avg_fit_overhead_time = []
            history_fit_phase_durations = []
            history_train_examples = []
            history_unique_train_examples = []
            history_current_examples = []
            history_replay_examples = []
            history_eval_examples = []
            if mode == 'centralized':
                display_name = LEARNING_TYPE_DISPLAY_NAMES['CL']
            elif mode == 'splitseq':
                display_name = LEARNING_TYPE_DISPLAY_NAMES['SL']
            else:
                display_name = f"{STRATEGY_DISPLAY_NAMES.get(mode, mode)} ({args.learning_type})"
            print(f"\n=== Starting Flower simulation: {display_name} Baseline ===")
            tracker = PerformanceTracker()
            sequential_group_size = effective_num_clients if (mode == 'splitseq' and args.comparison_profile == 'fair') else 1
            physical_num_rounds = args.rounds * sequential_group_size if sequential_group_size > 1 else args.rounds

            if uses_local_split:
                initial_model, back_model = build_split_model_pair()
                initial_model = initial_model.to(device)
                back_model = back_model.to(device)
                initial_params = [val.detach().cpu().numpy() for _, val in initial_model.state_dict().items()]
                initial_params.extend([val.detach().cpu().numpy() for _, val in back_model.state_dict().items()])
            elif uses_splitfed:
                initial_model, _ = build_split_model_pair()
                initial_model = initial_model.to(device)
                initial_params = [val.detach().cpu().numpy() for _, val in initial_model.state_dict().items()]
            else:
                initial_model = build_full_model().to(device)
                initial_params = [val.detach().cpu().numpy() for _, val in initial_model.state_dict().items()]
            initial_parameters = fl.common.ndarrays_to_parameters(initial_params)

            fraction_fit = (
                1.0
                if effective_num_clients <= 1
                else float(args.clients_per_round) / float(effective_num_clients)
            )

            common_params = {
                "fraction_fit": fraction_fit,
                "fraction_evaluate": 1.0,
                "min_fit_clients": 1 if mode == 'splitseq' else args.clients_per_round,
                "min_evaluate_clients": effective_num_clients,
                "min_available_clients": effective_num_clients,
                "on_fit_config_fn": tracker.start_fit,
                "fit_metrics_aggregation_fn": tracker.aggregate_fit_metrics,
                "evaluate_metrics_aggregation_fn": tracker.stop_evaluate,
                "initial_parameters": initial_parameters,
            }

            if is_splitfed_v1 and mode == 'fedavg':
                strategy = SplitFedV1Strategy(
                    server_actor_names=splitfed_server_actor_names,
                    server_actor_kwargs=splitfed_actor_kwargs,
                    server_parameters=[param.copy() for param in splitfed_initial_server_parameters],
                    **common_params,
                )
            elif is_splitfed_v2 and mode == 'fedavg':
                strategy = SplitFedV2Strategy(
                    server_actor_name=splitfed_shared_actor_name,
                    server_actor_kwargs=splitfed_actor_kwargs,
                    server_parameters=[param.copy() for param in splitfed_initial_server_parameters],
                    seed=args.seed,
                    **common_params,
                )
            elif mode == 'centralized' or mode == 'fedavg':
                strategy = fl.server.strategy.FedAvg(**common_params)
            elif mode == 'splitseq':
                strategy = SequentialRoundRobin(
                    evaluate_every_n_rounds=sequential_group_size,
                    **common_params,
                )
            elif mode == 'fedavgm':
                strategy = fl.server.strategy.FedAvgM(server_learning_rate=1.0, server_momentum=0.9, **common_params)
            elif mode == 'fedadam':
                strategy = fl.server.strategy.FedAdam(eta=0.01, eta_l=0.01, beta_1=0.9, beta_2=0.99, tau=0.01, **common_params)
            elif mode == 'fedprox':
                strategy = fl.server.strategy.FedProx(proximal_mu=0.1, **common_params)
            elif mode == 'fedmedian':
                strategy = fl.server.strategy.FedMedian(**common_params)
            elif mode == 'fedgsw':
                strategy = FedGSW(**common_params)
            elif mode == 'fedlaa':
                strategy = FedLAA(**common_params)
            elif mode == 'fedama':
                strategy = FedAMA(**common_params)
            elif mode == 'fedde':
                strategy = FedDE(**common_params)
            elif mode == 'fedsbfl':
                strategy = FedSBFL(**common_params)
            else:
                print(f"Skipping unknown baseline: {mode}")
                continue

            client_resources = {"num_cpus": 0.5}
            if device == "cuda":
                client_resources["num_gpus"] = 1.0 / max(effective_num_clients, 1)

            history = fl.simulation.start_simulation(
                client_fn=client_fn,
                num_clients=effective_num_clients,
                config=fl.server.ServerConfig(num_rounds=physical_num_rounds),
                strategy=strategy,
                client_resources=client_resources,
            )

            if "accuracy" in history.metrics_distributed:
                needs_collapse = (
                    sequential_group_size > 1
                    and len(history.metrics_distributed.get("accuracy", [])) > args.rounds
                )
                if needs_collapse:
                    collapsed = collapse_sequential_round_results(history, list(tracker.durations), sequential_group_size)
                    history_acc = collapsed["accuracy"]
                    durations_fixed = collapsed["durations"]
                    history_tflops = collapsed["tflops"]
                    history_avg_training_time = collapsed["avg_training_time_sec"]
                    history_avg_communication_time = collapsed["avg_communication_time_sec"]
                    history_avg_fit_time = collapsed["avg_fit_time_sec"]
                    history_avg_fit_overhead_time = collapsed["avg_fit_overhead_time_sec"]
                    history_fit_phase_durations = collapsed["fit_phase_duration_sec"]
                    history_train_examples = collapsed["train_examples"]
                    history_unique_train_examples = collapsed["unique_train_examples"]
                    history_current_examples = collapsed["current_examples"]
                    history_replay_examples = collapsed["replay_examples"]
                    history_eval_examples = collapsed["eval_examples"]
                else:
                    history_acc = [float(val) for _, val in history.metrics_distributed["accuracy"]]
                    durations_fixed = list(tracker.durations)
                    if len(durations_fixed) > 1:
                        durations_fixed[0] = float(np.mean(durations_fixed[1:]))
                    history_tflops = [float(val) for _, val in history.metrics_distributed.get("tflops", [])]
                    history_avg_training_time = [float(val) for _, val in history.metrics_distributed.get("avg_training_time_sec", [])]
                    history_avg_communication_time = [float(val) for _, val in history.metrics_distributed.get("avg_communication_time_sec", [])]
                    history_avg_fit_time = [float(val) for _, val in history.metrics_distributed.get("avg_fit_time_sec", [])]
                    history_avg_fit_overhead_time = [float(val) for _, val in history.metrics_distributed.get("avg_fit_overhead_time_sec", [])]
                    history_fit_phase_durations = [float(val) for _, val in history.metrics_distributed.get("fit_phase_duration_sec", [])]
                    history_train_examples = [float(val) for _, val in history.metrics_distributed.get("train_examples", [])]
                    history_unique_train_examples = [float(val) for _, val in history.metrics_distributed.get("unique_train_examples", [])]
                    history_current_examples = [float(val) for _, val in history.metrics_distributed.get("current_examples", [])]
                    history_replay_examples = [float(val) for _, val in history.metrics_distributed.get("replay_examples", [])]
                    history_eval_examples = [float(val) for _, val in history.metrics_distributed.get("eval_examples", [])]

                all_metric_results[display_name] = history_acc
                all_time_results[display_name] = durations_fixed
                if history_tflops:
                    all_tflops_results[display_name] = history_tflops
                if history_avg_training_time:
                    all_avg_training_time_results[display_name] = history_avg_training_time
                if history_avg_communication_time:
                    all_avg_communication_time_results[display_name] = history_avg_communication_time
                if history_fit_phase_durations:
                    all_fit_phase_duration_results[display_name] = history_fit_phase_durations

            csv_path = (
                f"csv/baseline_{file_mode_name(mode)}_"
                f"{model_name}_{args.dataset}_{effective_num_clients}Clients.csv"
            )
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'Round',
                    'Accuracy',
                    'Duration_Sec',
                    'Clients',
                    'Epochs',
                    'Model',
                    'Dataset',
                    'Data Distr. (Alpha)',
                    'Learning Type',
                    'TFLOPS',
                    'Avg_Training_Time_Sec',
                    'Avg_Communication_Time_Sec',
                    'Avg_Fit_Time_Sec',
                    'Avg_Fit_Overhead_Time_Sec',
                    'Fit_Phase_Duration_Sec',
                    'Train_Examples',
                    'Unique_Train_Examples',
                    'Current_Examples',
                    'Replay_Examples',
                    'Eval_Examples',
                ])
                for r in range(len(history_acc)):
                    dur = durations_fixed[r] if r < len(durations_fixed) else 0.0
                    tflops = history_tflops[r] if r < len(history_tflops) else 0.0
                    avg_training_time = history_avg_training_time[r] if r < len(history_avg_training_time) else 0.0
                    avg_communication_time = history_avg_communication_time[r] if r < len(history_avg_communication_time) else 0.0
                    avg_fit_time = history_avg_fit_time[r] if r < len(history_avg_fit_time) else 0.0
                    avg_fit_overhead_time = history_avg_fit_overhead_time[r] if r < len(history_avg_fit_overhead_time) else 0.0
                    fit_phase_duration = history_fit_phase_durations[r] if r < len(history_fit_phase_durations) else 0.0
                    train_examples = history_train_examples[r] if r < len(history_train_examples) else 0.0
                    unique_train_examples = history_unique_train_examples[r] if r < len(history_unique_train_examples) else 0.0
                    current_examples = history_current_examples[r] if r < len(history_current_examples) else 0.0
                    replay_examples = history_replay_examples[r] if r < len(history_replay_examples) else 0.0
                    eval_examples = history_eval_examples[r] if r < len(history_eval_examples) else 0.0
                    writer.writerow([
                        r + 1,
                        history_acc[r],
                        dur,
                        effective_num_clients,
                        args.epochs,
                        model_name,
                        args.dataset,
                        args.data_distr,
                        LEARNING_TYPE_DISPLAY_NAMES[args.learning_type],
                        tflops,
                        avg_training_time,
                        avg_communication_time,
                        avg_fit_time,
                        avg_fit_overhead_time,
                        fit_phase_duration,
                        train_examples,
                        unique_train_examples,
                        current_examples,
                        replay_examples,
                        eval_examples,
                    ])
            print(f"Log saved to {csv_path}")

        if all_metric_results:
            plt.figure(figsize=(10, 6))
            for display_name in all_metric_results.keys():
                plt.plot(range(1, len(all_metric_results[display_name]) + 1), all_metric_results[display_name], label=display_name, marker='o')
            plt.xlabel('Round')
            plt.ylabel('Testing Accuracy')
            plt.title(f'Testing Accuracy - {model_name} / {args.dataset} / {LEARNING_TYPE_DISPLAY_NAMES[args.learning_type]}')
            plt.legend()
            plt.grid(False)
            acc_filename = f"accuracy_{baseline_name_for_file}_{model_name}_{args.dataset}_{effective_num_clients}Clients.pdf"
            acc_plot_path = os.path.join(args.vis_dir, acc_filename)
            plt.savefig(acc_plot_path)
            plt.close()
            print(f"\nAccuracy plot saved to {acc_plot_path}")

            plt.figure(figsize=(10, 6))
            for display_name in all_time_results.keys():
                plt.plot(range(1, len(all_time_results[display_name]) + 1), all_time_results[display_name], label=display_name, marker='s')
            plt.xlabel('Round')
            plt.ylabel('Round Duration (seconds)')
            plt.title(f'Time per Round Comparison - {args.dataset} (Model: {model_name}, {LEARNING_TYPE_DISPLAY_NAMES[args.learning_type]})')
            plt.legend()
            plt.grid(False)
            time_filename = f"time_{baseline_name_for_file}_{model_name}_{args.dataset}_{effective_num_clients}Clients.pdf"
            time_plot_path = os.path.join(args.vis_dir, time_filename)
            plt.savefig(time_plot_path)
            plt.close()
            print(f"Time plot saved to {time_plot_path}")

            if all_tflops_results:
                plt.figure(figsize=(10, 6))
                for display_name in all_tflops_results.keys():
                    plt.plot(
                        range(1, len(all_tflops_results[display_name]) + 1),
                        all_tflops_results[display_name],
                        label=display_name,
                        marker='^',
                    )
                plt.xlabel('Round')
                plt.ylabel('Estimated Throughput (TFLOPS)')
                plt.title(f'TFLOPS per Round - {args.dataset} (Model: {model_name}, {LEARNING_TYPE_DISPLAY_NAMES[args.learning_type]})')
                plt.legend()
                plt.grid(False)
                tflops_filename = f"tflops_{baseline_name_for_file}_{model_name}_{args.dataset}_{effective_num_clients}Clients.pdf"
                tflops_plot_path = os.path.join(args.vis_dir, tflops_filename)
                plt.savefig(tflops_plot_path)
                plt.close()
                print(f"TFLOPS plot saved to {tflops_plot_path}")

            if all_avg_training_time_results:
                plt.figure(figsize=(10, 6))
                for display_name in all_avg_training_time_results.keys():
                    plt.plot(
                        range(1, len(all_avg_training_time_results[display_name]) + 1),
                        all_avg_training_time_results[display_name],
                        label=display_name,
                        marker='d',
                    )
                plt.xlabel('Round')
                plt.ylabel('Avg Training Time per Client (seconds)')
                plt.title(f'Average Client Training Time - {args.dataset} (Model: {model_name}, {LEARNING_TYPE_DISPLAY_NAMES[args.learning_type]})')
                plt.legend()
                plt.grid(False)
                training_time_filename = f"training_time_{baseline_name_for_file}_{model_name}_{args.dataset}_{effective_num_clients}Clients.pdf"
                training_time_plot_path = os.path.join(args.vis_dir, training_time_filename)
                plt.savefig(training_time_plot_path)
                plt.close()
                print(f"Training time plot saved to {training_time_plot_path}")

            if all_avg_communication_time_results:
                plt.figure(figsize=(10, 6))
                for display_name in all_avg_communication_time_results.keys():
                    plt.plot(
                        range(1, len(all_avg_communication_time_results[display_name]) + 1),
                        all_avg_communication_time_results[display_name],
                        label=display_name,
                        marker='x',
                    )
                plt.xlabel('Round')
                plt.ylabel('Avg Communication Time per Client (seconds)')
                plt.title(f'Average Client Communication Time - {args.dataset} (Model: {model_name}, {LEARNING_TYPE_DISPLAY_NAMES[args.learning_type]})')
                plt.legend()
                plt.grid(False)
                communication_time_filename = f"communication_time_{baseline_name_for_file}_{model_name}_{args.dataset}_{effective_num_clients}Clients.pdf"
                communication_time_plot_path = os.path.join(args.vis_dir, communication_time_filename)
                plt.savefig(communication_time_plot_path)
                plt.close()
                print(f"Communication time plot saved to {communication_time_plot_path}")

            if all_fit_phase_duration_results:
                plt.figure(figsize=(10, 6))
                for display_name in all_fit_phase_duration_results.keys():
                    plt.plot(
                        range(1, len(all_fit_phase_duration_results[display_name]) + 1),
                        all_fit_phase_duration_results[display_name],
                        label=display_name,
                        marker='v',
                    )
                plt.xlabel('Round')
                plt.ylabel('Fit Phase Duration (seconds)')
                plt.title(f'Fit Phase Duration - {args.dataset} (Model: {model_name}, {LEARNING_TYPE_DISPLAY_NAMES[args.learning_type]})')
                plt.legend()
                plt.grid(False)
                fit_phase_filename = f"fit_phase_time_{baseline_name_for_file}_{model_name}_{args.dataset}_{effective_num_clients}Clients.pdf"
                fit_phase_plot_path = os.path.join(args.vis_dir, fit_phase_filename)
                plt.savefig(fit_phase_plot_path)
                plt.close()
                print(f"Fit phase time plot saved to {fit_phase_plot_path}")

if __name__ == "__main__":
    main()
