import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

# ----------------------------------------------------------------
# 策略模块 1: SE Attention Block (注意力机制)
# 作用：自动学习每个通道的权重，抑制背景噪声，放大关键纹理
# ----------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: 压缩空间信息 [B, C, H, W] -> [B, C, 1, 1]
        y = self.avg_pool(x).view(b, c)
        # Excitation: 生成通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale: 将权重乘回原始特征
        return x * y.expand_as(x)

# ----------------------------------------------------------------
# 核心网络: ResNet18 + SE + Multi-scale Fusion
# ----------------------------------------------------------------
class AdvancedGarbageNet(nn.Module):
    def __init__(self, num_classes=12, dropout_rate=0.5):
        super(AdvancedGarbageNet, self).__init__()
        
        # 1. 加载骨干网络 (Backbone)
        print("正在加载 ResNet18 预训练权重...")
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 拆解 ResNet，方便提取中间层特征
        self.initial_layers = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
        )
        self.layer1 = resnet.layer1 # Output: 64 channels
        self.layer2 = resnet.layer2 # Output: 128 channels
        self.layer3 = resnet.layer3 # Output: 256 channels
        self.layer4 = resnet.layer4 # Output: 512 channels
        
        # 2. 策略模块插入: 为不同深度的特征层添加注意力机制
        self.se2 = SEBlock(channel=128)
        self.se3 = SEBlock(channel=256)
        self.se4 = SEBlock(channel=512)
        
        # 3. 策略模块: 多尺度特征融合 (Multi-scale Fusion)
        # 最终的特征维度 = layer2 + layer3 + layer4 = 128 + 256 + 512 = 896
        self.fusion_dim = 128 + 256 + 512
        
        # 4. 分类头 (Classifier)
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion_dim, 512),
            nn.BatchNorm1d(512),       # BN层加速收敛
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),  # Dropout 防止过拟合
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # --- 前向传播过程 ---
        
        # 基础层 (初始卷积 + 池化)
        x = self.initial_layers(x)
        x = self.layer1(x) # [B, 64, 56, 56] - 这一层特征太浅，通常不用来做融合
        
        # --- 提取多尺度特征 ---
        
        # Layer 2 (浅层语义/纹理)
        f2 = self.layer2(x)      # [B, 128, 28, 28]
        f2 = self.se2(f2)        # 加入注意力
        
        # Layer 3 (中层语义/部件)
        f3 = self.layer3(f2)     # [B, 256, 14, 14]
        f3 = self.se3(f3)        # 加入注意力
        
        # Layer 4 (深层语义/整体)
        f4 = self.layer4(f3)     # [B, 512, 7, 7]
        f4 = self.se4(f4)        # 加入注意力
        
        # --- 特征融合 (GAP + Concat) ---
        
        # 对三个尺度的特征分别进行全局平均池化 (Global Avg Pool) -> 变一维向量
        p2 = F.adaptive_avg_pool2d(f2, (1, 1)).view(f2.size(0), -1) # [B, 128]
        p3 = F.adaptive_avg_pool2d(f3, (1, 1)).view(f3.size(0), -1) # [B, 256]
        p4 = F.adaptive_avg_pool2d(f4, (1, 1)).view(f4.size(0), -1) # [B, 512]
        
        # 拼接 (Concatenate)
        fusion_feature = torch.cat((p2, p3, p4), dim=1) # [B, 896]
        
        # --- 分类 ---
        output = self.classifier(fusion_feature)
        
        return output

# ----------------------------------------------------------------
# 策略模块 3: Label Smoothing Loss (标签平滑损失)
# 作用：放在 model.py 里作为一个工具类供训练时调用
# ----------------------------------------------------------------
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        confidence = 1.0 - self.smoothing
        log_probs = F.log_softmax(pred, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

# --- 测试代码 ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 实例化模型
    model = AdvancedGarbageNet(num_classes=12).to(device)
    
    # 2. 模拟输入
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    # 3. 前向测试
    output = model(dummy_input)
    
    print("-" * 50)
    print("Advanced Architecture Build Successful!")
    print(f"Fusion Feature Dimension: {model.fusion_dim}") # 应该是 896
    print(f"Output Shape: {output.shape}") # 应该是 [2, 12]
    
    # 4. 测试 Loss
    criterion = LabelSmoothingCrossEntropy(smoothing=0.1)
    dummy_target = torch.tensor([0, 11]).to(device) # 假设两个标签
    loss = criterion(output, dummy_target)
    print(f"Label Smoothing Loss Test: {loss.item():.4f}")
    print("-" * 50)