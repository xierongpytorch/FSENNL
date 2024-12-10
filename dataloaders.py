import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset

from utils import get_device
from torchvision import transforms, datasets
import os
import cv2

BATCH_SIZE = 512
image_size = 32

train_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                         0.2023, 0.1994, 0.2010])
])

val_transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                         0.2023, 0.1994, 0.2010])
])

cifar_train = datasets.CIFAR10(
    root="datasets/cifar_train", train=True, transform=train_transform, download=True)
cifar_test = datasets.CIFAR10(
    root="datasets/cifar_test", train=False, transform=val_transform, download=True)

cifar_train_loader = DataLoader(
    cifar_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
cifar_test_loader = DataLoader(
    cifar_test, batch_size=BATCH_SIZE, shuffle=False, drop_last=True)




# 自定义数据集类
class CustomCIFAR10Dataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = os.listdir(image_dir)
        self.label_map = {
            'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4,
            'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9
        }

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_name)

        # 从文件名中提取标签
        label = None
        for label_name in self.label_map:
            if label_name in image_name.lower():  # 忽略大小写
                label = self.label_map[label_name]
                break

        # 确保找到了标签
        if label is None:
            raise ValueError(f"Image name {image_name} does not match any label.")

        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转换为RGB格式

        if self.transform:
            image = self.transform(image)

        return image, label


# 定义数据转换
custom_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((image_size, image_size)),  # Resize到32x32以符合CIFAR-10标准
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
])

# 创建自定义数据集
custom_train_dataset = CustomCIFAR10Dataset(
    image_dir="", transform=custom_transform)

# 封装数据加载器
custom_train_loader = DataLoader(
    custom_train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)