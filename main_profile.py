import warnings
import torch

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)
    device = torch.device("cuda")
    # choose your yaml file
    # model = YOLO("config/yolov8n-mamba-seg.yaml")
    model = YOLO("config/yolov8-contextMamba-seg.yaml")  # FYK_240616
    # model = YOLO("config/yolov8-seg-test.yaml")  # FYK_240616

    print("=========================================")
    model.info(detailed=True)
    try:
        model.profile(imgsz=[512, 512])
    except Exception as e:
        print(e)
        pass
    model.fuse()
