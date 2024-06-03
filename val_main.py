from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(model="best.pt")
    model.val(
        data="datastes/CAS_yolo_dataset/yolo.yaml",
        # batch=32,
        imgsz=512,
        split="val",
        save_hybrid=True,
        project="fastsam_output",
        name="240529_yolo8_CAS_all_val",
    )
