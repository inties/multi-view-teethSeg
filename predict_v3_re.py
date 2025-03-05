#修改版：使用flyby算法进行牙齿分割
import sys
import numpy as np
import time
import itk
import argparse
import os
import post_process
import fly_by_features as fbf
import torch
import vtk
from monai.networks.nets import UNet

parser = argparse.ArgumentParser(description='Predict with PyTorch model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--surf', type=str, help='Input surface mesh to label', required=True)
parser.add_argument('--model', type=str, help='Path to PyTorch .pth model', default="model_segmentation_02_02_unet_csv1.pth")
parser.add_argument('--dilate', type=int, help='Number of iterations to dilate the boundary', default=0)
parser.add_argument('--out', type=str, help='Output model with labels', default="out.vtk")
parser.add_argument('--resolution', type=int, help='Resolution of FlyBy images', default=320, choices=[320, 512])

args = parser.parse_args()

start_time = time.time()

# 读取和归一化表面网格
surf = fbf.ReadSurf(args.surf)
unit_surf = fbf.GetUnitSurf(surf)

# 创建二十面体和 FlyByGenerator
sphere = fbf.CreateIcosahedron(radius=2.75, sl=3)  # 92 个视点
flyby = fbf.FlyByGenerator(sphere, resolution=args.resolution, visualize=False, use_z=True, split_z=True)

surf_actor = fbf.GetNormalsActor(unit_surf)
flyby.addActor(surf_actor)
print("FlyBy features ...")
img_np = flyby.getFlyBy(save_images=True)  # (92, resolution, resolution, 4)
print("Number of sphere points:", sphere.GetNumberOfPoints())
print("Input shape before prediction:", img_np.shape)
flyby.removeActors()

point_id_actor = fbf.GetPointIdMapActor(unit_surf)
flyby_features = fbf.FlyByGenerator(sphere, args.resolution, visualize=False)
flyby_features.addActor(point_id_actor)
print("FlyBy features point id map ...")
img_point_id_map_np = flyby_features.getFlyBy()  # (92, resolution, resolution, 3)
img_point_id_map_np = img_point_id_map_np.reshape((-1, 3))  # (92*resolution*resolution, 3)
flyby_features.removeActors()

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
img_predict_np = np.argmax(img_predict_np, axis=1).reshape(-1)  # (92*resolution*resolution,)

# 标签投票
prediction_array_count = np.zeros([surf.GetNumberOfPoints(), 34])
for point_id_rgb, prediction in zip(img_point_id_map_np, img_predict_np):
    r, g, b = point_id_rgb
    point_id = int(b * 255 * 255 + g * 255 + r - 1)
    if point_id >= 0 and point_id < surf.GetNumberOfPoints():
        prediction_array_count[point_id][int(prediction)] += 1

real_labels = vtk.vtkIntArray()
real_labels.SetNumberOfComponents(1)
real_labels.SetNumberOfTuples(surf.GetNumberOfPoints())
real_labels.SetName("RegionId")
real_labels.Fill(0)
for pointId, prediction in enumerate(prediction_array_count):
    if np.max(prediction) > 0:
        label = np.argmax(prediction)
        real_labels.SetTuple(pointId, (label,))
surf.GetPointData().AddArray(real_labels)

# 保存和后处理
outfilename_pre = os.path.splitext(args.out)[0] + "_pre.vtk"
print("Writing:", outfilename_pre)
polydatawriter = vtk.vtkPolyDataWriter()
polydatawriter.SetFileName(outfilename_pre)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

if args.dilate:
    print("Dilate...")
    post_process.DilateLabel(surf, real_labels, 3, iterations=args.dilate)

labels_range = np.zeros(2)
real_labels.GetRange(labels_range)
for label in range(int(labels_range[0]), int(labels_range[1]) + 1):
    print("Removing islands:", label)
    post_process.RemoveIslands(surf, real_labels, label, 200)

out_filename = os.path.splitext(args.out)[0] + "_islands.vtk"
print("Writing:", out_filename)
polydatawriter.SetFileName(out_filename)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

print("Relabel...")
post_process.ReLabel(surf, real_labels, 3, -1)

print("Connectivity...")
post_process.ConnectivityLabeling(surf, real_labels, 2, 2)

out_filename = os.path.splitext(args.out)[0] + "_connectivity.vtk"
print("Writing:", out_filename)
polydatawriter.SetFileName(out_filename)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

print("Eroding...")
post_process.ErodeLabel(surf, real_labels, -1, ignore_label=0)

print("Writing:", args.out)
polydatawriter.SetFileName(args.out)
polydatawriter.SetInputData(surf)
polydatawriter.Write()

teeth_surf = post_process.Threshold(surf, real_labels, 2, 999999)
outfilename_teeth = os.path.splitext(args.out)[0] + "_teeth.vtk"
print("Writing:", outfilename_teeth)
polydatawriter.SetFileName(outfilename_teeth)
polydatawriter.SetInputData(teeth_surf)
polydatawriter.Write()

gum_surf = post_process.Threshold(surf, real_labels, 0, 1)
outfilename_gum = os.path.splitext(args.out)[0] + "_gum.vtk"
print("Writing:", outfilename_gum)
polydatawriter.SetFileName(outfilename_gum)
polydatawriter.SetInputData(gum_surf)
polydatawriter.Write()

end_time = time.time()
print("Prediction time took:", (end_time - start_time), "seconds")