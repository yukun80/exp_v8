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

    model = YOLO(model="config/yolov8x-seg.yaml")  # build a new model from YAML
    # model = YOLO(model="weights/yolov8x-seg.pt")  # load a pre-existing model from PyTorch checkpoint(recommended for training)
    # model.info(detailed=True)
    model.load("weights/yolov8x-seg.pt")
    model.train(
        data="datastes/CAS_yolo_dataset/yolo.yaml",
        epochs=300,
        batch=128,
        imgsz=512,
        save=True,
        save_period=20,
        device="0,1,2,3",
        overlap_mask=False,
        project="fastsam_output",
        name="240531_yolo8_CAS_all",
        workers=32,
        cache=True,
        seed=3407,
        close_mosaic=20,
    )
