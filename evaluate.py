import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns # 如果没有安装，请 pip install seaborn
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

# 引用你的模型定义
from model import AdvancedGarbageNet
from data_loader import get_data_loaders

# ================= 配置 =================
DATA_DIR = r'E:\ai_math_hw\garbage_classification\data'
MODEL_PATH = 'best_model.pth'
HISTORY_PATH = 'training_history.json'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =======================================

def plot_training_history(history_path):
    """绘制 Loss 和 Accuracy 曲线"""
    if not os.path.exists(history_path):
        print("未找到训练历史文件，跳过绘图。")
        return

    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_acc'], 'b-', label='Training Acc')
    plt.plot(epochs, history['val_acc'], 'r-', label='Validation Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('training_curves.png') # 保存图片用于报告
    print("曲线图已保存为 training_curves.png")
    plt.show()

def evaluate_model(model, dataloader, classes):
    """在测试集上评估，计算混淆矩阵"""
    model.eval()
    all_preds = []
    all_labels = []

    print("正在测试集上进行最终评估...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 1. 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("混淆矩阵已保存为 confusion_matrix.png")
    plt.show()

    # 2. 分类报告 (Precision, Recall, F1)
    report = classification_report(all_labels, all_preds, target_names=classes)
    print("\nClassification Report:")
    print(report)
    
    # 保存报告到文本文件
    with open('result_report.txt', 'w') as f:
        f.write(report)

if __name__ == '__main__':
    # 1. 绘制曲线
    plot_training_history(HISTORY_PATH)

    # 2. 加载数据
    _, _, test_loader, classes = get_data_loaders(DATA_DIR, batch_size=16)

    # 3. 加载最佳模型
    print(f"加载模型权重: {MODEL_PATH}")
    model = AdvancedGarbageNet(num_classes=len(classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)

    # 4. 执行评估
    evaluate_model(model, test_loader, classes)