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
    # model = YOLO("config/yolov8x-seg.yaml")
    # model = YOLO("config/yolov8m-mamba-seg.yaml")
    # model = YOLO("config/Mamba-YOLO-L.yaml")  # FYK_240616
    # model = YOLO("config/yolov8x-MambaContextGLSAFocal.yaml")
    model = YOLO("config/yolov8l-MambaGLSAFocal.yaml")
    # model = YOLO("config/yolov8l-FocalMamba.yaml")

    print("=========================================")
    model.info(detailed=True)
    # 创建一个形状为 (1, 3, 512, 512) 的全零张量
    test_input = torch.zeros(8, 3, 512, 512).to(device)
    try:
        model.profile(test_input)
    except Exception as e:
        print(e)
        pass
    model.fuse()
