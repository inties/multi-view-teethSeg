import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import argparse

def visualize_vtk_with_labels(vtk_file_path):
    # 1. 读取 VTK 文件
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_file_path)
    reader.Update()
    polydata = reader.GetOutput()

    # 2. 获取点数据和 UniversalID 标签
    point_data = polydata.GetPointData()
    labels_array = point_data.GetArray("UniversalID")
    if labels_array is None:
        print("错误：未找到'UniversalID'标签数组")
        return
    
    labels = vtk_to_numpy(labels_array)
    unique_labels = np.unique(labels)
    print(f"找到的唯一标签值：{unique_labels}")

    # 3. 创建颜色映射
    colors = vtk.vtkUnsignedCharArray()
    colors.SetNumberOfComponents(3)
    colors.SetName("Colors")

    # 定义颜色表（0-33 的 RGB 值）
    color_map = {
        0: [255, 0, 0],    # 红色
        1: [0, 255, 0],    # 绿色
        2: [0, 0, 255],    # 蓝色
        3: [255, 255, 0],  # 黄色
        4: [255, 0, 255],  # 品红
        5: [0, 255, 255],  # 青色
        6: [128, 0, 0],    # 深红
        7: [0, 128, 0],    # 深绿
        8: [0, 0, 128],    # 深蓝
        9: [128, 128, 0],  # 橄榄色
        10: [128, 0, 128], # 紫色
        11: [0, 128, 128], #  teal
        12: [192, 192, 192], # 浅灰
        13: [255, 165, 0],  # 橙色
        14: [255, 192, 203], # 粉色
        15: [75, 0, 130],   # 靛蓝
        16: [245, 245, 220], # 米色
        17: [139, 69, 19],  # 棕色
        18: [240, 230, 140], # 卡其色
        19: [64, 224, 208], #  turquoise
        20: [173, 255, 47], # 黄绿
        21: [220, 20, 60],  # 猩红
        22: [135, 206, 235], # 天蓝
        23: [147, 112, 219], # 中紫
        24: [255, 215, 0],  # 金色
        25: [0, 191, 255],  # 深天蓝
        26: [218, 165, 32], # 金棕色
        27: [255, 20, 147], # 深粉
        28: [34, 139, 34],  # 森林绿
        29: [138, 43, 226], # 蓝紫
        30: [255, 99, 71],  # 番茄红
        31: [32, 178, 170], # 浅海绿
        32: [240, 128, 128], # 浅珊瑚
        33: [106, 90, 205]  # 石板蓝
    }

    # 为每个顶点分配颜色
    for i in range(polydata.GetNumberOfPoints()):
        label = labels[i]
        if label not in color_map:
            print(f"警告：标签 {label} 超出预定义范围，使用默认颜色")
            colors.InsertNextTuple([255, 255, 255])  # 超出范围用白色
        else:
            colors.InsertNextTuple(color_map[label])

    polydata.GetPointData().AddArray(colors)
    polydata.GetPointData().SetActiveScalars("Colors")

    # 4. 创建渲染管道（显示多边形网格）
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointData()

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetPointSize(5)  # 调整点的大小
    actor.GetProperty().EdgeVisibilityOn()  # 显示网格边框
    actor.GetProperty().SetEdgeColor(0, 0, 0)  # 边框颜色为黑色

    # 5. 设置渲染器和窗口
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.4)  # 背景颜色

    # 6. 添加交互功能
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    interactor.Initialize()
    render_window.Render()
    interactor.Start()

def main():
    parser = argparse.ArgumentParser(description="渲染 VTK 文件并根据 UniversalID 显示不同颜色")
    parser.add_argument("vtk_file", type=str, help="输入的 .vtk 文件路径")
    args = parser.parse_args()
    visualize_vtk_with_labels(args.vtk_file)

if __name__ == "__main__":
    main()