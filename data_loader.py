import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_data_loaders(data_dir, batch_size=32, split_ratio=[0.8, 0.1, 0.1], seed=42):
    """
    Args:
        data_dir (str): 数据集根目录 (包含12个子文件夹的路径)
        batch_size (int): 批大小
        split_ratio (list): [训练集, 验证集, 测试集] 的比例，和必须为1
        seed (int): 随机种子，保证每次切分结果一致 (为了复现性)
    
    Returns:
        train_loader, val_loader, test_loader, classes
    """
    
    # 1. 定义图像预处理 (Transforms)
    # -----------------------------------------------------------
    # 训练集：包含数据增强 (Augmentation)
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),       # 随机裁剪，增加难度
        transforms.RandomHorizontalFlip(),       # 水平翻转
        transforms.RandomVerticalFlip(),         # 垂直翻转 (垃圾通常无方向)
        transforms.RandomRotation(20),           # 随机旋转
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1), # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 验证集/测试集：仅做基础处理，不做增强
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),           # 直接Resize或CenterCrop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 2. 加载数据集 (关键技巧：加载两次)
    # -----------------------------------------------------------
    # 为了让训练集和验证集应用不同的Transform，我们需要实例化两个Dataset对象
    full_dataset_train = datasets.ImageFolder(root=data_dir, transform=train_transform)
    full_dataset_val   = datasets.ImageFolder(root=data_dir, transform=val_test_transform)
    
    classes = full_dataset_train.classes
    print(f"检测到类别 ({len(classes)}): {classes}")

    # 3. 划分数据集索引
    # -----------------------------------------------------------
    num_samples = len(full_dataset_train)
    indices = list(range(num_samples))
    
    # 设置随机种子并打乱索引
    np.random.seed(seed)
    np.random.shuffle(indices)

    train_split = int(np.floor(split_ratio[0] * num_samples))
    val_split = int(np.floor(split_ratio[1] * num_samples))
    
    train_idx = indices[:train_split]
    val_idx = indices[train_split : train_split + val_split]
    test_idx = indices[train_split + val_split:]

    print(f"数据划分完成 -> 训练集: {len(train_idx)}, 验证集: {len(val_idx)}, 测试集: {len(test_idx)}")

    # 4. 创建 Subset 和 DataLoader
    # -----------------------------------------------------------
    # 注意：train_subset 使用带增强的 dataset，val/test_subset 使用不带增强的 dataset
    train_subset = Subset(full_dataset_train, train_idx)
    val_subset   = Subset(full_dataset_val, val_idx)
    test_subset  = Subset(full_dataset_val, test_idx)

    # num_workers 设置为 2 或 4 可以加速数据读取，Windows下如果报错请改为 0
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader, classes

# 测试代码块 (当你直接运行此文件时会执行)
if __name__ == '__main__':
    DATA_DIR = r'E:\ai_math_hw\garbage_classification\data' 
    
    if os.path.exists(DATA_DIR):
        print(f"正在读取路径: {DATA_DIR}")
        # 这里为了测试，batch_size 设小一点，真正训练时可以设为 16 或 32
        train_loader, val_loader, test_loader, classes = get_data_loaders(DATA_DIR, batch_size=16)
        
        print(f"成功识别到 {len(classes)} 个类别：{classes}")
        
        # 打印一个 Batch 看看形状
        images, labels = next(iter(train_loader))
        print(f"图片 Batch 形状: {images.shape}") # 应该是 [4, 3, 224, 224]
        print(f"标签 Batch 形状: {labels.shape}")
    else:
        print(f"错误：路径 {DATA_DIR} 不存在，请检查拼写。")

    #查查我的小显卡在不在
    # print(f"PyTorch Version: {torch.__version__}")
    # print(f"CUDA Available: {torch.cuda.is_available()}")

    # if torch.cuda.is_available():
    #     print(f"Device Name: {torch.cuda.get_device_name(0)}")
    # else:
    #     print("警告：未检测到显卡，将使用 CPU 训练 (速度会很慢)")


# 数据来源  感谢  https://www.kaggle.com/datasets/mostafaabla/garbage-classification/data

