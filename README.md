# VisionTrack AI
## Real-Time Object Detection and Tracking System

VisionTrack AI is a beginner-friendly Python computer vision project that detects objects in webcam streams or uploaded videos and tracks them across frames.

### Features
- Webcam detection
- Uploaded video processing (MP4, AVI, MOV, MKV)
- Pretrained YOLOv8 nano object detection
- ByteTrack multi-object tracking
- Bounding boxes, class names and confidence scores
- Tracking IDs when assigned by the tracker
- Detection statistics and processed MP4 download
- CPU inference; no GPU or custom training required

### Technologies
Python, Streamlit, OpenCV, Ultralytics YOLOv8, ByteTrack, Pandas.

### Project structure
```text
VisionTrack_AI/
├── app.py
├── detector.py
├── tracker.py
├── video_processor.py
├── requirements.txt
├── README.md
├── .gitignore
├── uploads/
└── output/
```

### Windows / VS Code installation
1. Install Python 3.10 or 3.11.
2. Open this folder in VS Code.
3. Open Terminal > New Terminal.
4. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
5. Activate it:
   ```bash
   venv\\Scripts\\activate
   ```
6. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Run:
   ```bash
   python -m streamlit run app.py
   ```
8. Open the local URL shown in the terminal, usually http://localhost:8501.

### How to use
1. Select **Upload Video** or **Webcam**.
2. For video mode, upload an MP4/AVI/MOV/MKV clip.
3. Set the detection confidence threshold.
4. Click **Start Detection**.
5. View the annotated video frame, labels, confidence scores and tracking IDs.
6. Click **Stop Detection** when desired.
7. Download the processed MP4.

### Model
The project uses the pretrained `yolov8n.pt` model. Ultralytics downloads the weights automatically on first run, so an internet connection is needed for initial setup. No training from scratch is required.

### Notes and limitations
- CPU inference can be slower than GPU inference. Start with short, 480p or 720p videos.
- The app explicitly selects CPU inference for compatibility.
- The webcam is the camera attached to the computer running Streamlit.
- Tracking IDs are temporary tracker identifiers, not identities of real people.
- Detection accuracy depends on lighting, resolution, camera angle and object size.
- This project writes output to `output/processed_video.mp4`; a later run may overwrite that file.

### CodeAlpha Internship
Artificial Intelligence Internship — Task 4: Real-Time Object Detection and Tracking System.
