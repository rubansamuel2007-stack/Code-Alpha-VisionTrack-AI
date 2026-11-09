import cv2
import threading
from pathlib import Path
from collections import Counter

from detector import ObjectDetector

def process_video(source, job, stop_event, confidence=0.35,
                  is_webcam=False, output_path="output/processed_video.mp4"):
    cap = None
    writer = None

    try:
        with job["lock"]:
            job["status"] = "Loading YOLOv8 model..."

        detector = ObjectDetector()

        if is_webcam:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(str(source))

        if not cap.isOpened():
            raise RuntimeError("Could not open the webcam or uploaded video. Check the file or camera permissions.")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0 or fps > 120:
            fps = 20.0

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if width <= 0 or height <= 0:
            raise RuntimeError("The video has invalid dimensions.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create the output MP4 file.")

        frame_number = 0
        total_detections = 0
        class_counts = Counter()
        unique_track_ids = set()

        with job["lock"]:
            job["status"] = "Detection running..."

        while not stop_event.is_set():
            ok, frame = cap.read()
            if not ok:
                break

            annotated, detections = detector.process_frame(frame, confidence)
            writer.write(annotated)
            frame_number += 1
            total_detections += len(detections)

            for item in detections:
                class_counts[item["Object"]] += 1
                track_id = item["Tracking ID"]
                if isinstance(track_id, int):
                    unique_track_ids.add(track_id)

            ok, encoded = cv2.imencode(".jpg", annotated)
            if ok:
                with job["lock"]:
                    job["frame"] = encoded.tobytes()
                    job["frame_number"] = frame_number
                    job["total_detections"] = total_detections
                    job["unique_objects"] = len(unique_track_ids)
                    job["class_counts"] = dict(class_counts)
                    job["last_detections"] = detections
                    job["status"] = "Detection running..."

        final_status = "Stopped by user" if stop_event.is_set() else "Processing completed"
        with job["lock"]:
            job["status"] = final_status
            job["output_path"] = str(output)
            job["completed"] = True

    except Exception as exc:
        with job["lock"]:
            job["status"] = f"Error: {exc}"
            job["error"] = str(exc)
            job["completed"] = True

    finally:
        if cap is not None:
            cap.release()
        if writer is not None:
            writer.release()
        with job["lock"]:
            job["running"] = False
