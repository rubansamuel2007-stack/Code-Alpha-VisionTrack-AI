def extract_tracking_data(result):
    detections = []
    boxes = result.boxes

    if boxes is None or len(boxes) == 0:
        return detections

    for index in range(len(boxes)):
        class_id = int(boxes.cls[index].item())
        confidence = float(boxes.conf[index].item())
        object_name = result.names[class_id]

        track_id = None
        if boxes.id is not None:
            track_id = int(boxes.id[index].item())

        detections.append({
            "Object": object_name,
            "Confidence (%)": round(confidence * 100, 2),
            "Tracking ID": track_id if track_id is not None else "Not assigned"
        })

    return detections
