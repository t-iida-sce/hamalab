from typing import Type

from torch.utils.data import Dataset
from torchvision.datasets import MNIST, FashionMNIST, KMNIST
from qmnist import QMNIST

__all__ = [
    "get_mnist_dataset",
]


# noinspection SpellCheckingInspection
def get_mnist_dataset(dataset_name: str) -> Type[Dataset]:
    dataset_name = dataset_name.casefold()
    if not dataset_name or dataset_name == "MNIST".casefold():
        return MNIST
    elif dataset_name == "QMNIST".casefold():
        return QMNIST
    elif dataset_name == "FashionMNIST".casefold():
        return FashionMNIST
    elif dataset_name == "KMNIST".casefold():
        return KMNIST
    else:
        raise ValueError(f"Unknown dataset {dataset_name=}")
