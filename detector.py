from ultralytics import YOLO
from tracker import extract_tracking_data

class ObjectDetector:
    def __init__(self, model_path="yolov8n.pt"):
        # Ultralytics downloads yolov8n.pt automatically on first use.
        self.model = YOLO(model_path)

    def process_frame(self, frame, confidence=0.35):
        results = self.model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=confidence,
            imgsz=640,
            device="cpu",
            verbose=False
        )
        result = results[0]
        annotated = result.plot()
        detections = extract_tracking_data(result)
        return annotated, detections
