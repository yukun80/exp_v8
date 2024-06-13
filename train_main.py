import sys
import os
from ultralytics import YOLO

os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    # 假设 `ultralytics` 目录位于项目根目录下
    project_root = "/share/home/hanbochen22_ucas/fyk/ultralytics-main"
    ultralytics_path = os.path.join(project_root, "ultralytics")
    if ultralytics_path not in sys.path:
        sys.path.append(ultralytics_path)

    model = YOLO(model="config/yolov8x-C2f-LVMB.yaml")
    model.train(
        # 233333
        data="datastes/yolo_dataset_sub/yolo_sub.yaml",
        epochs=1,
        batch=4,
        imgsz=512,
        save=True,
        save_period=1,
        device="0",
        overlap_mask=False,
        project="fastsam_output",
        name="240611_y8_CAS_ba_Context",
        workers=8,
        cache=False,
        seed=3407,
        close_mosaic=20,
        patience=60,
    )
    print("Done!")
