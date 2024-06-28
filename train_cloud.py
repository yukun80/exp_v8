import sys
import os
from ultralytics import YOLO

os.environ["WANDB_MODE"] = "offline"

if __name__ == "__main__":
    # 假设 `ultralytics` 目录位于项目根目录下
    project_root = "/share/home/hanbochen22_ucas/fyk/ultralytics-cloud"
    ultralytics_path = os.path.join(project_root, "ultralytics")
    if ultralytics_path not in sys.path:
        sys.path.append(ultralytics_path)

    # build a new model from YAML
    model = YOLO(model="config/yolov8l-MambaGLSAFocal.yaml")
    # model.load("weights/yolov9e-seg.pt")
    model.train(
        # 233333
        data="datasets/CAS_balance_yolo_datasets/yolo_balance.yaml",
        epochs=500,
        batch=64,
        imgsz=512,
        save=True,
        save_period=50,
        device="0,1,2,3",
        overlap_mask=False,
        project="fastsam_output",
        name="240628_MambaGLSAFocal",
        workers=128,
        cache=False,
        seed=3407,
        close_mosaic=400,
        patience=80,
    )
