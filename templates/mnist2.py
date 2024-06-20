#!/usr/bin/env python3
import argparse
from typing import Optional

import mnist_dataset
import lightning as pl
import torch.optim
import torchmetrics
from lightning import Trainer
from lightning.pytorch.utilities.types import *
from lightning.pytorch.callbacks import EarlyStopping
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class MNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        activation = nn.PReLU  # Change here to `nn.ReLU` to use ReLU.
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
        self.soft_max = nn.Softmax(dim=1)
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.Accuracy("multiclass", num_classes=10)
        self.f1_score = torchmetrics.F1Score("multiclass", num_classes=10, average="macro")

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

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def _do_prediction(self, batch):
        x, y = batch
        y_prediction = self.soft_max(self.forward(x))
        loss = self.loss_function(y_prediction, y)
        self.accuracy(y_prediction, y)
        self.f1_score(y_prediction, y)
        return loss

    def training_step(self, batch, batch_index) -> STEP_OUTPUT:
        loss = self._do_prediction(batch)
        self.log("train_accuracy", self.accuracy, sync_dist=True, prog_bar=True)
        self.log("train_f1score", self.f1_score, sync_dist=True, prog_bar=True)
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        loss = self._do_prediction(batch)
        self.log("val_accuracy", self.accuracy, sync_dist=True)
        self.log("val_f1score", self.f1_score, sync_dist=True)
        self.log("val_loss", loss, sync_dist=True)
        return loss

    def test_step(self, batch, batch_index) -> Optional[STEP_OUTPUT]:
        loss = self._do_prediction(batch)
        self.log("test_accuracy", self.accuracy, sync_dist=True)
        self.log("test_f1score", self.f1_score, sync_dist=True)
        self.log("test_loss", loss, sync_dist=True)
        return loss


class MNISTLoader(pl.LightningDataModule):
    _train_data: Dataset
    _validation_data: Dataset
    _test_data: Dataset

    def __init__(self, mini_batch: int, dataset_name: str, validation_ratio: float = 0.2):
        super().__init__()
        self.mini_batch = mini_batch
        self.dataset_name = dataset_name
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
        self.validation_ratio = validation_ratio
        self.root_path = "./data"

    def prepare_data(self, stage: Optional[str] = None) -> None:
        dataset_class = mnist_dataset.get_mnist_dataset(self.dataset_name)
        # Just download data.
        dataset_class(root=self.root_path, train=True, download=True, transform=self.transform)
        dataset_class(root=self.root_path, train=False, download=True, transform=self.transform)
        return

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_class = mnist_dataset.get_mnist_dataset(self.dataset_name)
        train_all = dataset_class(root=self.root_path, train=True, download=True, transform=self.transform)
        train_size = len(train_all)
        validation_size = int(train_size * self.validation_ratio)
        self._train_data, self._validation_data = torch.utils.data.random_split(train_all, [
            train_size - validation_size, validation_size])
        self._test_data = dataset_class(root=self.root_path, train=False, download=True, transform=self.transform)
        return

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self._train_data, batch_size=self.mini_batch, num_workers=4, persistent_workers=True)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self._validation_data, batch_size=self.mini_batch, num_workers=4, persistent_workers=True)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self._test_data, batch_size=self.mini_batch, num_workers=4, persistent_workers=True)


def main():
    torch.set_float32_matmul_precision('high')
    parser = argparse.ArgumentParser(description="Run MNIST.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--batch", type=int, help="batch size", default=512)
    parser.add_argument("-e", "--epoch", type=int, help="epoch count", default=100)
    parser.add_argument("-d", "--dataset", type=str, help="dataset name", default="MNIST")
    parser.add_argument("--device", type=str, help="device", choices=["gpu", "cpu"], default="gpu")
    args = parser.parse_args()
    device_count = max(1, torch.cuda.device_count())
    print(f"Batch Size={args.batch}, Dataset={args.dataset}, Epoch={args.epoch}")

    model = MNISTModel()
    loader = MNISTLoader(args.batch, args.dataset)
    early_stopping = EarlyStopping("val_loss", min_delta=0, patience=5, mode="min")
    trainer = Trainer(max_epochs=args.epoch, accelerator=args.device, devices=device_count, callbacks=[early_stopping])
    trainer.fit(model, loader)
    trainer.validate(model, loader)
    trainer.test(model, loader)


if __name__ == '__main__':
    main()
