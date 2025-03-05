import sys
import numpy as np
import time
import argparse
import torch
from monai.networks.nets import UNet
import fly_by_features as fbf
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Predict with PyTorch model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
    parser.add_argument('--model', type=str, help='Path to PyTorch .pth model', default="model_segmentation_02_02_unet_csv1.pth")
    parser.add_argument('--resolution', type=int, help='Resolution of FlyBy images', default=320, choices=[320, 512])
    parser.add_argument('--output_dir', type=str, help='Directory to save output images', default="output_images")
    return parser.parse_args()

def generate_color_map(num_classes=34):
    """生成一个颜色映射表，将 0-33 的标签映射到 RGB 颜色"""
    colors = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        # 使用简单的映射方案，确保颜色分布均匀且可区分
        r = (i * 7) % 256  # 红色通道递增，循环
        g = (i * 11) % 256  # 绿色通道递增，步长不同避免重叠
        b = (i * 13) % 256  # 蓝色通道递增，步长不同
        colors[i] = [r, g, b]
    return colors

def main():
    args = parse_args()

    start_time = time.time()

    # 读取和归一化表面网格
    surf = fbf.ReadSurf(args.surf)
    unit_surf = fbf.GetUnitSurf(surf)

    # 创建二十面体和 FlyByGenerator
    sphere = fbf.CreateIcosahedron(radius=2.75, sl=2)  # 92 个视点
    flyby = fbf.FlyByGenerator(sphere, resolution=args.resolution, visualize=False, use_z=True, split_z=True)

    surf_actor = fbf.GetNormalsActor(unit_surf)
    flyby.addActor(surf_actor)
    print("FlyBy features ...")
    img_np = flyby.getFlyBy(save_images=True)  # (92, resolution, resolution, 4)
    print("Number of sphere points:", sphere.GetNumberOfPoints())
    print("Input shape before prediction:", img_np.shape)
    flyby.removeActors()

    # 加载 PyTorch 模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(
        spatial_dims=2,
        in_channels=4,
        out_channels=34,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
    ).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    # 预测
    print("Predict ...")
    with torch.no_grad():
        img_tensor = torch.from_numpy(img_np).float().to(device)  # (92, resolution, resolution, 4)
        img_tensor = img_tensor.permute(0, 3, 1, 2)  # (92, 4, resolution, resolution)
        img_predict_np = model(img_tensor).cpu().numpy()  # (92, 34, resolution, resolution)
    print("Output shape after prediction:", img_predict_np.shape)

    # 获取预测标签
    predicted_labels = np.argmax(img_predict_np, axis=1)  # (92, resolution, resolution)

    # 生成颜色映射表
    color_map = generate_color_map(num_classes=34)

    # 创建输出目录
    import os
    os.makedirs(args.output_dir, exist_ok=True)

    # 转换为 RGB 图像并保存
    print("Converting predictions to RGB images ...")
    for i in range(42):
        # 将标签映射到 RGB 颜色
        rgb_image = color_map[predicted_labels[i]]  # (resolution, resolution, 3)
        # 转换为 PIL 图像并保存
        img = Image.fromarray(rgb_image, mode='RGB')
        output_path = os.path.join(args.output_dir, f"viewpoint_{i:03d}.png")
        img.save(output_path)
        print(f"Saved image {i+1}/92: {output_path}")

    end_time = time.time()
    print("Prediction and conversion time took:", (end_time - start_time), "seconds")

if __name__ == "__main__":
    main()