import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

"""
1. 修改任务名称
2. 修改测试数据集路径
3. 修改模型权重路径
"""

if __name__ == "__main__":
    """设置任务名和路径"""
    task_name = "CAS_yolo9_train_r0_2"
    source_dir, mask_dir, label_dir, yolo_lable_dir = (
        # os.path.join("datastes", "CAS_balance_yolo_datasets", "images", "test"),
        # os.path.join("datastes", "CAS_yolo_test"),
        os.path.join("datastes", "CAS_yolo_train"),
        os.path.join("results", task_name, "_mask"),
        os.path.join("results", task_name, "_label"),
        os.path.join("results", task_name, "_yolo"),
    )
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    os.makedirs(yolo_lable_dir, exist_ok=True)

    # Get a list of image files in the source directory
    image_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    # model = YOLO(
    #     model="fastsam_output/240608_yolo9_CAS_balance/weights/best.pt"
    # )
    model = YOLO(model="weights/yolo9_r0_best.pt")

    """对图像进行预测，返回结果列表"""
    # Process results generator
    for i, image_file in enumerate(tqdm(image_files)):
        # Get the base filename without extension
        base_filename = os.path.splitext(os.path.basename(image_files[i]))[0]
        for result in model.predict(
            source=image_file,
            conf=0.1,
            stream=True,
            save=True,
            show_labels=False,
            show_conf=False,
            show_boxes=False,
            save_conf=False,
        ):
            # 如果有检测结果，才进行处理
            if result.masks is not None:
                """在结果路径下把文件名保存到txt中"""
                with open(
                    os.path.join("results", task_name, "result_list.txt"),
                    "a",
                ) as f:
                    f.write(f"{base_filename}\n")
                """提取掩膜图像"""
                masks = result.masks.data
                boxes = result.boxes.data
                # extract classes, 每个维度的含义参考Boxes类的定义，通常包括（x中心，y中心，宽度，高度，置信度，类别）。
                clss = boxes[:, -1]
                # get indices of results where class is 0 (people in COCO)
                landslide_indices = torch.where(clss == 0)
                # use these indices to extract the relevant masks
                landslide_masks = masks[landslide_indices]
                """保存为掩膜图像，可用于定量分析"""
                landslide_mask = torch.any(landslide_masks, dim=0).byte().cpu()
                # 将掩膜图像保存到tif文件
                cv2.imwrite(
                    str(os.path.join(mask_dir, f"{base_filename}.tif")),
                    landslide_mask.cpu().numpy(),
                )
                """保存为RGB图像，用于查看"""
                landslide_label = np.stack(
                    [landslide_mask] * 3, axis=-1
                )  # Convert to 3 channels
                landslide_label = (landslide_label * 255).astype(
                    np.uint8
                )  # Convert to uint8
                # Save to TIFF file
                cv2.imwrite(
                    str(os.path.join(label_dir, f"{base_filename}.tif")),
                    landslide_label,
                )
                """获取多边形端点坐标，保存为yolo格式的标签文件"""
                polygon_coords_pixel = (
                    result.masks.xyn
                )  # 获取像素坐标的多边形端点
                with open(
                    os.path.join(yolo_lable_dir, f"{base_filename}.txt"),
                    "w",
                ) as f:
                    for poly in polygon_coords_pixel:
                        f.write(f"0 ")
                        for x, y in poly:
                            f.write(f"{x:.6f} {y:.6f} ")
                        f.write("\n")
    print("Done!")
