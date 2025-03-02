import torch
from monai.networks.nets import UNet
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "E:\\GitHub\\Fly-by-CNN-3.0\\07-21-22_val-loss0.169.pth"  # 替换为你的 .pth 文件路径

model = UNet(
    spatial_dims=2,
    in_channels=4,
    out_channels=34,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
).to(device)

model.load_state_dict(torch.load(model_path, map_location=device))
print("Model loaded successfully.")
model.eval()

input_shape = (92, 320, 320, 4)  # predict_test.py 输出
dummy_input = torch.randn(input_shape).to(device)
dummy_input = dummy_input.permute(0, 3, 1, 2)  # (92, 4, 512, 512)

try:
    with torch.no_grad():
        output = model(dummy_input)
    print("Model accepts the input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    expected_classes = 34
    output_channels = output.shape[1]
    if output_channels == expected_classes:
        print(f"Output channels match expected classes: {output_channels}")
    else:
        print(f"Output channels ({output_channels}) do not match expected classes ({expected_classes})")
except Exception as e:
    print(f"Model cannot accept the input shape {dummy_input.shape}: {e}")