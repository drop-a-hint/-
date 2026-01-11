import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import torch.nn.functional as F
import copy
import time
import pandas as pd
from tqdm import tqdm

# 复用你现有的数据加载器
from data_loader import get_data_loaders

# ================= 配置区域 =================
DATA_DIR = r'E:\ai_math_hw\garbage_classification\data' # 确认路径
BATCH_SIZE = 16
ABLATION_EPOCHS = 8  # 消融实验不需要跑25轮，8-10轮足够看清趋势，节省时间
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 1. 定义可配置的模型类 =================
class FlexibleGarbageNet(nn.Module):
    def __init__(self, num_classes=12, use_se=False, use_multiscale=False):
        super(FlexibleGarbageNet, self).__init__()
        self.use_se = use_se
        self.use_multiscale = use_multiscale
        
        # 加载骨干
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.initial_layers = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        
        # SE 模块定义 (仅当 use_se=True 时会被用到)
        if self.use_se:
            self.se2 = self._make_se_block(128)
            self.se3 = self._make_se_block(256)
            self.se4 = self._make_se_block(512)
            
        # 分类头定义
        if self.use_multiscale:
            # 融合维度: 128 + 256 + 512 = 896
            self.fc = nn.Linear(896, num_classes)
        else:
            # 原始 ResNet 维度: 512
            self.fc = nn.Linear(512, num_classes)

    def _make_se_block(self, channel, reduction=16):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.layer1(x)
        
        # Layer 2
        f2 = self.layer2(x)
        if self.use_se: 
            b, c, _, _ = f2.size()
            w = self.se2(f2).view(b, c, 1, 1)
            f2 = f2 * w
            
        # Layer 3
        f3 = self.layer3(f2)
        if self.use_se:
            b, c, _, _ = f3.size()
            w = self.se3(f3).view(b, c, 1, 1)
            f3 = f3 * w
            
        # Layer 4
        f4 = self.layer4(f3)
        if self.use_se:
            b, c, _, _ = f4.size()
            w = self.se4(f4).view(b, c, 1, 1)
            f4 = f4 * w
        
        if self.use_multiscale:
            # 多尺度融合模式
            p2 = F.adaptive_avg_pool2d(f2, (1, 1)).view(f2.size(0), -1)
            p3 = F.adaptive_avg_pool2d(f3, (1, 1)).view(f3.size(0), -1)
            p4 = F.adaptive_avg_pool2d(f4, (1, 1)).view(f4.size(0), -1)
            out = torch.cat((p2, p3, p4), dim=1)
        else:
            # 原始 ResNet 模式 (只用最后一层)
            out = F.adaptive_avg_pool2d(f4, (1, 1)).view(f4.size(0), -1)
            
        return self.fc(out)

# ================= 2. 训练与评估函数 =================
def run_experiment(name, use_se, use_multiscale, use_labelsmoothing, train_loader, val_loader):
    print(f"\n>>> 正在运行实验: [{name}]")
    print(f"    配置: SE={use_se}, MultiScale={use_multiscale}, LabelSmoothing={use_labelsmoothing}")
    
    # 初始化模型
    model = FlexibleGarbageNet(num_classes=12, use_se=use_se, use_multiscale=use_multiscale).to(DEVICE)
    
    # 定义 Loss
    if use_labelsmoothing:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    else:
        criterion = nn.CrossEntropyLoss() # 普通 CrossEntropy
        
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    
    best_acc = 0.0
    
    # 简单的训练循环
    for epoch in range(ABLATION_EPOCHS):
        # Train
        model.train()
        for inputs, labels in tqdm(train_loader, desc=f"Ep {epoch+1}/{ABLATION_EPOCHS}", leave=False):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
        # Val
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        val_acc = correct / total
        if val_acc > best_acc:
            best_acc = val_acc
            
    print(f"    [{name}] 最终最高验证准确率: {best_acc:.4%}")
    
    # 清理显存，防止 OOM
    del model
    torch.cuda.empty_cache()
    
    return best_acc

# ================= 3. 主程序 =================
if __name__ == '__main__':
    print("=== 开始消融实验 (Ablation Study) ===")
    print(f"每组实验运行 {ABLATION_EPOCHS} 轮，请耐心等待...")
    
    # 加载数据
    train_loader, val_loader, _, _ = get_data_loaders(DATA_DIR, batch_size=BATCH_SIZE)
    
    # 定义实验列表
    # 格式: (实验名, use_se, use_multiscale, use_labelsmoothing)
    experiments = [
        ("1. Baseline (ResNet18)", False, False, False),
        ("2. ResNet + SE",         True,  False, False),
        ("3. ResNet + MultiScale", False, True,  False),
        ("4. Ours (Full Method)",  True,  True,  True)
    ]
    
    results = []
    
    for exp_name, se, ms, ls in experiments:
        acc = run_experiment(exp_name, se, ms, ls, train_loader, val_loader)
        results.append({
            "Experiment": exp_name,
            "SE Attention": "✅" if se else "❌",
            "Multi-Scale": "✅" if ms else "❌",
            "Label Smooth": "✅" if ls else "❌",
            "Best Val Acc": f"{acc:.2%}"
        })
        
    # 输出结果表格
    print("\n" + "="*60)
    print("消融实验最终结果汇总")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_markdown(index=False))
    
    # 保存到 CSV 文件
    df.to_csv("ablation_results.csv", index=False)
    print("\n结果已保存至 ablation_results.csv")