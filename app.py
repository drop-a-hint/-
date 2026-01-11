import streamlit as st
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import pandas as pd
import os

# 引用你的模型定义
from model import AdvancedGarbageNet

# ================= 配置参数 =================
MODEL_PATH = 'best_model.pth' # 确保这里是你的模型路径
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 固定的类别列表 (必须与训练时一致)
CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]

# ================= 核心函数 =================

@st.cache_resource # 缓存模型，避免每次预测都重新加载
def load_model():
    # 实例化你定义的网络结构
    model = AdvancedGarbageNet(num_classes=len(CLASSES))
    
    # 加载权重
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        print("模型加载成功！")
        return model
    else:
        st.error(f"找不到模型文件: {MODEL_PATH}")
        return None

def process_image(image):
    """图片预处理，与训练时的验证集处理一致"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) # 增加 Batch 维度 [1, 3, 224, 224]

def predict(model, image_tensor):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        # 使用 Softmax 获取概率分布
        probs = F.softmax(outputs, dim=1)
    return probs.cpu().numpy()[0]

# ================= 界面构建 =================

def main():
    # 1. 页面标题和侧边栏
    st.set_page_config(page_title="垃圾智能分类系统", page_icon="♻️")
    st.title("♻️ 深度学习垃圾分类系统")
    st.markdown("### 基于 ResNet18-SE-MultiScale 架构")
    
    st.sidebar.title("项目信息")
    st.sidebar.info(
        """
        - **模型**: ResNet18 + SE Attention + Multi-scale
        - **准确率**: 97% (Test Set)
        - **数据集**: Garbage Classification (12 classes)
        - **开发者**: [拖拉机拉飞机]
        """
    )

    # 2. 加载模型
    model = load_model()
    if model is None:
        return

    # 3. 图片上传区
    uploaded_file = st.file_uploader("请上传一张垃圾图片...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # 显示上传的图片
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(image, caption='上传的图片', use_container_width=True)
        
        with col2:
            st.markdown("### 识别结果")
            if st.button('开始识别'):
                # 预处理
                img_tensor = process_image(image)
                
                # 预测
                probs = predict(model, img_tensor)
                
                # 获取 Top-1 结果
                top_idx = np.argmax(probs)
                top_class = CLASSES[top_idx]
                top_prob = probs[top_idx]
                
                # 展示主要结果
                st.success(f"**预测类别:** {top_class.upper()}")
                st.metric(label="置信度", value=f"{top_prob:.2%}")
                
                # 展示详细概率分布图 (Top 5)
                st.markdown("#### 概率分布 (Top 5)")
                
                # 整理数据用于绘图
                top5_indices = probs.argsort()[-5:][::-1]
                top5_probs = probs[top5_indices]
                top5_classes = [CLASSES[i] for i in top5_indices]
                
                df = pd.DataFrame({
                    'Class': top5_classes,
                    'Probability': top5_probs
                })
                
                st.bar_chart(df.set_index('Class'))

if __name__ == '__main__':
    main()