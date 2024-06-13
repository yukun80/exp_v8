import warnings
import torch

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    print(torch.cuda.is_available())
    print(torch.__version__)
    print(torch.version.cuda)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # choose your yaml file
    model = YOLO("config/yolov8x-C2f-LVMB.yaml").to(device)
    model.info(detailed=True)
    try:
        model.profile(imgsz=[512, 512])
    except Exception as e:
        print(e)
        pass
    model.fuse()
