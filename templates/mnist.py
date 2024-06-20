#!/usr/bin/env python3
import argparse
import os
import sys
import time

# Graph
import matplotlib.pyplot as plt
# Torch
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchinfo
import torchvision
from matplotlib.ticker import MultipleLocator
from torch import nn
from torchvision import transforms
# Utility
from tqdm import trange, tqdm
import mnist_dataset

N = 30  # Train iteration count
TEST_EPOCH = 5  # Evaluate test data for each TEST_EPOCH epochs.


def main(device: torch.device, mini_batch: int, is_gpu: bool, dataset_name: str) -> None:
    workers = 4 if is_gpu else 0
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset_class = mnist_dataset.get_mnist_dataset(dataset_name)

    train = dataset_class(
        root="./data", train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=mini_batch, shuffle=True, num_workers=workers, pin_memory=False
    )
    test = dataset_class(
        root="./data", train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test, batch_size=mini_batch, shuffle=False, num_workers=workers, pin_memory=False
    )
    train_and_evaluate(device, is_gpu, train_loader, test_loader, dataset_name)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        activation = nn.GELU  # Change here to `nn.ReLU` to use ReLU.
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, (3, 3)),  # 28x28x1 -> 26x26x32
            activation(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, (3, 3)),  # 26x26x32 -> 24x24x64
            activation(),
            nn.MaxPool2d(2, 2),  # 24x24x64 -> 12x12x64
        )
        self.dropout1 = nn.Dropout2d()
        self.fc1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12 * 12 * 64, 256),
            activation()
        )
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc2 = nn.Sequential(
            nn.Linear(256, 10),
            activation(),
        )

    def feature(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.dropout2(x)
        return x

    def forward(self, x):
        x = self.feature(x)
        x = self.fc2(x)
        return x


@torch.no_grad()
def evaluate(device: torch.device, net: Net, is_gpu: bool,
             loss_func, data_loader, get_loss=False):
    total, correct, loss = 0, 0, 0
    net.eval()
    p = tqdm(data_loader) if not get_loss else data_loader
    for inputs, labels in p:
        inputs = inputs.to(device, non_blocking=is_gpu)
        labels = labels.to(device, non_blocking=is_gpu)
        outputs = net(inputs)
        _, prediction = torch.max(outputs, 1)
        total += outputs.size(0)
        correct += (labels == prediction).sum().item()
        if get_loss:
            loss += loss_func(outputs, labels).item()
    if get_loss:
        return total, correct, loss / total
    else:
        return total, correct


def _print(content, title: str):
    print("*" * 10 + " " + title + " " + "*" * 10)
    print(content)
    print("*" * (20 + len(title) + 2))


def train_and_evaluate(device: torch.device, is_gpu: bool,
                       train_loader: torch.utils.data.DataLoader,
                       test_loader: torch.utils.data.DataLoader,
                       dataset_name: str):
    net = Net().to(device, )
    _print(net, "Network")
    torchinfo.summary(net, input_size=(10, 1, 28, 28), device=device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters())

    loss_log, accuracy_log = [], []
    test_epochs, test_accuracy_log, test_loss_log = [], [], []
    started = time.time()
    for epoch in trange(N):
        rl = 0.0
        _total, _correct = 0, 0
        net.train()
        for i, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss: torch.Tensor = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                _, prediction = torch.max(outputs, 1)
                _total += outputs.size(0)
                _correct += (labels == prediction).sum().item()

            rl += loss.item()
        loss_log.append(rl)
        accuracy_log.append(_correct / _total)
        if (epoch + 1) % TEST_EPOCH == 0 or epoch < 5:
            test_total, test_correct, test_loss = evaluate(
                device, net, is_gpu, loss_func, test_loader, True)
            test_accuracy_log.append(test_correct / test_total)
            test_loss_log.append(test_loss)
            test_epochs.append(epoch)
            tqdm.write(f"[epoch={epoch + 1:02}] train loss={rl:.3f} train correct={_correct} "
                       f"test={test_loss:.3f} test correct={test_correct} total={_total}")
    took = time.time() - started
    _print(title="Result", content=f"Learning {N} steps took {took}, average={took / N}.")

    make_graph(N, accuracy_log, device, loss_log,
               test_accuracy_log, test_epochs, test_loss_log, dataset_name)
    started = time.time()
    train_total, train_correct = evaluate(
        device, net, is_gpu, loss_func, train_loader)
    test_total, test_correct = evaluate(
        device, net, is_gpu, loss_func, test_loader)
    _print(title="Evaluation",
           content=f"Evaluation took {time.time() - started}.\n"
                   f"Train Accuracy {train_correct / train_total:.3f} ({train_correct=}, {train_total=})\n"
                   f"Test Accuracy {test_correct / test_total:.3f} ({test_correct=}, {test_total=})",
           )
    return


def make_graph(epochs, accuracy_log, device, loss_log, test_accuracy_log, test_epochs, test_loss_log,
               dataset_name: str):
    dataset_name = dataset_name or 'MNIST'
    fig: plt.Figure = plt.figure(figsize=(12, 9))
    ax1: plt.Axes = fig.add_subplot(1, 1, 1)
    ax2: plt.Axes = ax1.twinx()
    ax1.plot(np.arange(epochs) + 1, loss_log, label="TRAIN Loss")
    ax1.plot(np.array(test_epochs) + 1, test_loss_log, label="TEST Loss", ls="dashed")
    ax1.set_ylabel("Loss")
    ax1.set_xlabel("Epoch")
    ax1.get_xaxis().set_major_locator(MultipleLocator(5))
    ax1.get_xaxis().set_minor_locator(MultipleLocator(1))
    ax1.set_xlim(0, epochs)
    ax2.plot(np.arange(epochs) + 1, accuracy_log,
             label="TRAIN Accuracy", color="green")
    ax2.plot(np.array(test_epochs) + 1, test_accuracy_log, label="TEST Accuracy",
             color="purple", ls="dashed")
    ax2.set_ylim(min(min(accuracy_log), min(test_accuracy_log)) * 0.95, 1)
    ax2.set_ylabel("Accuracy")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc='right')
    ax1.grid()
    ax1.set_title(f"Learning Loss and Accuracy of {dataset_name} on {device}")
    fig.savefig(f"{dataset_name}_train.pdf", dpi=300, bbox_inches='tight', pad_inches=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run MNIST.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "device", choices=["gpu", "cpu"], help="device where this program works", )
    parser.add_argument("-b", "--batch", type=int, help="batch size", default=512)
    parser.add_argument("-d", "--dataset", type=str, help="dataset name", default="MNIST")
    args = parser.parse_args()

    is_cuda_available = torch.cuda.is_available()
    _is_gpu = args.device == "gpu"
    if _is_gpu and is_cuda_available:
        print("GPU is supported and selected.")
        _d = torch.device("cuda")
    elif _is_gpu and not is_cuda_available:
        print("GPU is NOT supported. Using CPU.")
        _d = torch.device("cpu")
    else:
        if is_cuda_available:
            print("GPU is supported but CPU is manually selected.")
        else:
            print("GPU is NOT supported. CPU is selected.")
        _d = torch.device("cpu")
        torch.set_num_threads(os.cpu_count())
        torch.set_num_interop_threads(os.cpu_count())
        _print(torch.__config__.parallel_info(), "Parallel Info")

    print(f"Device={_d}, Batch Size={args.batch}, Dataset={args.dataset}")
    main(_d, args.batch, _is_gpu, args.dataset)
