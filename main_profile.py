import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    model = YOLO('config/yolov8-seg-ContextGuideFPN.yaml')
    model.info(detailed=True)
    try:
        model.profile(imgsz=[512, 512])
    except Exception as e:
        print(e)
        pass
    model.fuse()