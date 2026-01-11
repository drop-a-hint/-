import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
import os
import matplotlib.pyplot as plt
from tqdm import tqdm  

# 导入我们可以自定义的模块
from data_loader import get_data_loaders
from model import AdvancedGarbageNet, LabelSmoothingCrossEntropy

# ==========================================
# 配置参数 (Hyper-parameters)
# ==========================================
DATA_DIR = r'E:\ai_math_hw\garbage_classification\data' # 你的数据路径
BATCH_SIZE = 16          # 16?32?(显存不够)
NUM_EPOCHS = 25          # 迁移学习通常 20-30 轮即可收敛
LEARNING_RATE = 1e-4     # 微调(Fine-tuning)学习率要小，不要用 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 Epoch"""
    model.train() # 切换到训练模式 (启用 Dropout, BN 更新)
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    # 使用 tqdm 显示进度条
    pbar = tqdm(dataloader, desc='Training', leave=False)
    
    for inputs, labels in pbar:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # 1. 清空梯度
        optimizer.zero_grad()

        # 2. 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # 3. 反向传播与优化
        loss.backward()
        optimizer.step()

        # 4. 统计指标
        _, preds = torch.max(outputs, 1)
        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += torch.sum(preds == labels.data)
        total_samples += batch_size
        
        # 更新进度条显示的当前 Loss
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def validate(model, dataloader, criterion, device):
    """验证集评估"""
    model.eval() # 切换到评估模式 (冻结 BN, 禁用 Dropout)
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    
    with torch.no_grad(): # 验证阶段不需要计算梯度，节省显存
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, preds = torch.max(outputs, 1)
            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += torch.sum(preds == labels.data)
            total_samples += batch_size

    epoch_loss = running_loss / total_samples
    epoch_acc = running_corrects.double() / total_samples
    
    return epoch_loss, epoch_acc.item()

def main():
    print(f"Using Device: {DEVICE}")
    
    # 1. 准备数据
    train_loader, val_loader, test_loader, classes = get_data_loaders(
        DATA_DIR, batch_size=BATCH_SIZE
    )
    
    # 2. 初始化模型
    model = AdvancedGarbageNet(num_classes=len(classes)).to(DEVICE)
    
    # 3. 定义损失函数 (使用我们自定义的 Label Smoothing)
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    
    # 4. 定义优化器 (AdamW 是目前最稳健的选择)
    # weight_decay 是正则化项，防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    # 5. 学习率调度器 (CosineAnnealing)
    # 让学习率随 Epoch 变化，最后降低到 1e-6
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    # 6. 开始训练循环
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    start_time = time.time()
    
    print(f"开始训练... 总轮数: {NUM_EPOCHS}")
    print("-" * 60)

    for epoch in range(NUM_EPOCHS):
        # --- Training ---
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # --- Validation ---
        val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
        
        # --- Scheduler Step ---
        scheduler.step()
        
        # --- 记录历史 ---
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # --- 打印本轮结果 ---
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")

        # --- 保存最佳模型 ---
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'best_model.pth')
            print(">>> 发现新高分模型，已保存！")
        
        print("-" * 60)

    total_time = time.time() - start_time
    print(f'训练完成！总耗时: {total_time // 60:.0f}m {total_time % 60:.0f}s')
    print(f'最佳验证集准确率: {best_acc:.4f}')

    # 7. 保存训练历史数据 (用于后续画图)
    import json
    with open('training_history.json', 'w') as f:
        json.dump(history, f)
    print("训练历史已保存至 training_history.json")

if __name__ == '__main__':
    # 这里的 if 判断是为了防止 Windows 下多进程 DataLoader 报错
    main()