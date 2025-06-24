from ultralytics import YOLO
import time

def download_models():
    # List of model names to download
    models = [
        # YOLOv5 detection
        'yolov5n.pt', 'yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt',
        # # YOLOv5 segmentation
        # 'yolov5n-seg.pt', 'yolov5s-seg.pt', 'yolov5m-seg.pt', 'yolov5l-seg.pt', 'yolov5x-seg.pt',
        # YOLOv8 detection
        'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt',
        # # YOLOv8 segmentation
        # 'yolov8n-seg.pt', 'yolov8s-seg.pt', 'yolov8m-seg.pt', 'yolov8l-seg.pt', 'yolov8x-seg.pt',
        # YOLOv11 detection
        'yolo11n.pt', 'yolo11s.pt', 'yolo11m.pt', 'yolo11l.pt', 'yolo11x.pt',
        # # YOLOv11 segmentation
        # 'yolo11n-seg.pt', 'yolo11s-seg.pt', 'yolo11m-seg.pt', 'yolo11l-seg.pt', 'yolo11x-seg.pt',
        # # YOLOv5 classification (if exists)
        # 'yolov5n-cls.pt', 'yolov5s-cls.pt', 'yolov5m-cls.pt', 'yolov5l-cls.pt', 'yolov5x-cls.pt',
        # # YOLOv8 classification
        # 'yolov8n-cls.pt', 'yolov8s-cls.pt', 'yolov8m-cls.pt', 'yolov8l-cls.pt', 'yolov8x-cls.pt',
        # # YOLOv11 classification
        # 'yolo11n-cls.pt', 'yolo11s-cls.pt', 'yolo11m-cls.pt', 'yolo11l-cls.pt', 'yolo11x-cls.pt',
    ]

    for model_name in models:
        try:
            model = YOLO("weights/"+model_name)
            print(f"成功下载 {model_name}")
        except Exception as e:
            print(f"下载 {model_name} 失败: {e}")
        time.sleep(1)  # 可选的延迟，避免对服务器造成过大压力

if __name__ == "__main__":
    print("开始下载模型...")
    download_models()
    print("下载完成。")

model=YOLO("/weights/yolov5/yolov5nu.pt")