import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO


if __name__ == "__main__":
    task_name = "CAS_yolo8_test"
    source_dir, mask_dir, label_dir = (
        os.path.join("datastes", task_name),  # source_dir: 图片文件夹路径
        os.path.join("results", task_name, "_mask"),
        os.path.join("results", task_name, "_label"),
    )
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    # Get a list of image files in the source directory
    image_files = [
        os.path.join(source_dir, f)
        for f in os.listdir(source_dir)
        if f.endswith((".jpg", ".png", ".jpeg", ".bmp", ".tif", ".tiff"))
    ]

    model = YOLO(model="weights/best.pt")

    """对图像进行预测，返回结果列表"""
    results = model.predict(
        source=image_files,
        conf=0.5,
        save=True,
        stream=True,
        show_labels=False,
        show_conf=False,
        show_boxes=False,
        save_conf=False,
    )

    # Process results generator
    for i, result in enumerate(results):
        # Get the base filename without extension
        base_filename = os.path.basename(image_files[i]).split(".")[0]
        # Construct the save path
        if result.masks is not None:
            masks = result.masks.data
            boxes = result.boxes.data
            # extract classes, 每个维度的含义参考Boxes类的定义，通常包括（x中心，y中心，宽度，高度，置信度，类别）。
            clss = boxes[:, -1]
            # get indices of results where class is 0 (people in COCO)
            landslide_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            landslide_masks = masks[landslide_indices]
            # scale for visualizing results
            landslide_mask = torch.any(landslide_masks, dim=0).int() * 255
            # save to file
            cv2.imwrite(
                str(os.path.join(mask_dir, f"{base_filename}_mask.jpg")),
                landslide_mask.cpu().numpy(),
            )

            polygon_conf = boxes[:, -2]
            print(f"==================Image {i} confidences: {polygon_conf}")

            # 获取多边形端点坐标
            polygon_coords_pixel = result.masks.xyn  # 获取像素坐标的多边形端点
            print(len(polygon_coords_pixel))
            # 输出坐标信息
            print("Polygon coordinates in pixel units:")
            for poly in polygon_coords_pixel:
                print(poly)
            # 将类别信息和坐标信息按照yolo格式保存到图片同名的txt文件中
            with open(
                os.path.join(label_dir, f"{base_filename}.txt"),
                "w",
            ) as f:
                for poly in polygon_coords_pixel:
                    f.write(f"0 ")
                    for x, y in poly:
                        f.write(f"{x:.6f} {y:.6f} ")
                    f.write("\n")
    print("Done!")
