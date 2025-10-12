# utils/data_utils.py
# PyTorch의 torchvision 라이브러리를 사용하여, MNIST, Fashion-MNIST, Kuzushiji-MNIST 중 지정된 이미지 데이터셋을 불러와
# 학습용(training)과 검증용(validation)으로 나눈 뒤, PyTorch의 `DataLoader` 형식으로 만들어 반환하는 함수

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import torch

def load_data(dataset_name, n_visible):
    """
    Args:
        dataset_name : 데이터셋 이름
        n_visible : 가시 유닛의 수
    """
    transform = transforms.Compose([transforms.ToTensor()])
    # transforms.ToTensor()는 이미지를 PyTorch 텐서로 변환하고, 픽셀 값을 0~255에서 0.0~1.0 사이로 정규화하는 역할
    
    if dataset_name == 'MNIST':
        dataset = datasets.MNIST('./DQA2/data', train=True, download=True, transform=transform)
    elif dataset_name == 'fMNIST':
        dataset = datasets.FashionMNIST('./DQA2/data', train=True, download=False, transform=transform)
    elif dataset_name == 'kMNIST':
        dataset = datasets.KMNIST('./DQA2/data', train=True, download=False, transform=transform)
    else:
        raise ValueError("Unsupported dataset")

    train_dataset, val_dataset = random_split(dataset, [50000, 10000])
    train_loader = DataLoader(train_dataset, batch_size=50000, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=10000, shuffle=False)

    return train_loader, val_loader
