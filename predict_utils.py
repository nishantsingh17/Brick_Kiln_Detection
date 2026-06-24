from ultralytics import YOLO, SAM


class BrickKilnDetector:
    def __init__(self, yolo_model_path, sam_model_path=None):
        self.yolo = YOLO(yolo_model_path)
        self.sam = SAM(sam_model_path) if sam_model_path else None
