import torch
import cv2
import numpy as np
import os
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 导入你的模型
from model import AdvancedGarbageNet

# ================= 配置区域 =================
# 1. 模型路径
MODEL_PATH = 'best_model.pth'

# 2. 你想测试的图片路径 (建议挑几张不同类别的)
# 请修改为你电脑上存在的真实图片路径！！！
IMG_PATH = r'E:\ai_math_hw\garbage_classification\data\clothes\clothes15.jpg' 

# 3. 类别名称 (用于显示预测结果)
CLASSES = [
    'battery', 'biological', 'brown-glass', 'cardboard', 'clothes', 
    'green-glass', 'metal', 'paper', 'plastic', 'shoes', 'trash', 'white-glass'
]
# ===========================================

def get_input_transform():
    """定义预处理 (必须与训练时一致)"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def run_gradcam():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")

    # 1. 加载模型
    print("正在加载模型...")
    model = AdvancedGarbageNet(num_classes=12)
    
    if os.path.exists(MODEL_PATH):
        # 这里的 map_location 保证即使在 CPU 上也能加载 GPU 训练的权重
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"错误: 找不到模型文件 {MODEL_PATH}")
        return

    model.to(device)
    model.eval()

    # 2. 选定目标层 (Target Layer)
    # 对于 ResNet 架构，通常看 layer4 的最后一层卷积
    # 这一层包含了最丰富的语义信息 (形状、纹理)
    target_layers = [model.layer4[-1]]

    # 3. 读取并处理图片
    if not os.path.exists(IMG_PATH):
        print(f"错误: 找不到图片 {IMG_PATH}")
        return

    # 打开图片并转为 RGB
    img_pil = Image.open(IMG_PATH).convert('RGB')
    img_pil = img_pil.resize((224, 224)) # 统一大小
    
    # 准备 visualization 用的背景图 (归一化到 0-1 的 numpy 数组)
    rgb_img = np.array(img_pil, dtype=np.float32) / 255.0
    
    # 准备模型输入的 tensor
    transform = get_input_transform()
    input_tensor = transform(img_pil).unsqueeze(0).to(device) # [1, 3, 224, 224]

    # 4. 先做一次预测，看看模型认为它是什么
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASSES[pred_idx]
        conf = probs[0][pred_idx].item()
        print(f"模型预测结果: {pred_class} (置信度: {conf:.2%})")

    # 5. 初始化 Grad-CAM
    cam = GradCAM(model=model, target_layers=target_layers)

    # 我们让 Grad-CAM 解释 "为什么模型预测它是 pred_idx 这个类"
    targets = [ClassifierOutputTarget(pred_idx)]

    # 生成热力图 (Grayscale)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    grayscale_cam = grayscale_cam[0, :] # 取第一张图的结果

    # 6. 将热力图叠加到原图上
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    # 7. 保存结果
    save_name = f"gradcam_{pred_class}.jpg"
    Image.fromarray(visualization).save(save_name)
    print(f"热力图已保存为: {save_name}")
    print("红色区域表示模型最关注的地方，蓝色表示不关注的地方。")

if __name__ == '__main__':
    run_gradcam()