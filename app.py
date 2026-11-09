import streamlit as st
import threading
from pathlib import Path
import pandas as pd

from video_processor import process_video

st.set_page_config(page_title="VisionTrack AI", page_icon="🎯", layout="wide")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #10182c, #172d4a); color: white; }
h1, h2, h3, p, label { color: white; }
.main-title { text-align:center; font-size:42px; font-weight:800; color:#52b6ff; }
.subtitle { text-align:center; color:#cbd9ee; font-size:17px; }
</style>
""", unsafe_allow_html=True)

if "job" not in st.session_state:
    st.session_state.job = {
        "lock": threading.Lock(),
        "stop_event": threading.Event(),
        "thread": None,
        "running": False,
        "completed": False,
        "status": "Ready",
        "frame": None,
        "frame_number": 0,
        "total_detections": 0,
        "unique_objects": 0,
        "class_counts": {},
        "last_detections": [],
        "output_path": None,
        "error": None,
    }

job = st.session_state.job

st.markdown('<div class="main-title">🎯 VISIONTRACK AI</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Real-Time Object Detection and Tracking System</div>', unsafe_allow_html=True)
st.divider()

with st.sidebar:
    st.title("🎯 VisionTrack AI")
    st.write("AI Object Detection Dashboard")
    st.divider()
    st.subheader("Technologies")
    st.write("Python • OpenCV • YOLOv8 • ByteTrack • Streamlit")
    st.info("Uses pretrained YOLOv8 nano weights. No custom training is required.")

st.header("📹 Select Video Source")
source_type = st.radio(
    "Choose input source", ["Upload Video", "Webcam"],
    horizontal=True, disabled=job["running"]
)

uploaded_file = None
if source_type == "Upload Video":
    uploaded_file = st.file_uploader(
        "Upload a video", type=["mp4", "avi", "mov", "mkv"],
        disabled=job["running"]
    )

confidence = st.slider(
    "Detection confidence", 0.10, 0.90, 0.35, 0.05,
    disabled=job["running"]
)

can_start = not job["running"] and (source_type == "Webcam" or uploaded_file is not None)
if st.button("🚀 Start Detection", use_container_width=True, disabled=not can_start):
    Path("uploads").mkdir(exist_ok=True)
    Path("output").mkdir(exist_ok=True)

    job["stop_event"] = threading.Event()
    with job["lock"]:
        job.update({
            "running": True, "completed": False, "status": "Starting...",
            "frame": None, "frame_number": 0, "total_detections": 0,
            "unique_objects": 0, "class_counts": {}, "last_detections": [],
            "output_path": None, "error": None
        })

    if source_type == "Webcam":
        source, is_webcam = 0, True
    else:
        safe_name = Path(uploaded_file.name).name
        source_path = Path("uploads") / safe_name
        source_path.write_bytes(uploaded_file.getbuffer())
        source, is_webcam = str(source_path), False

    thread = threading.Thread(
        target=process_video,
        args=(source, job, job["stop_event"], confidence, is_webcam, "output/processed_video.mp4"),
        daemon=True
    )
    job["thread"] = thread
    thread.start()
    st.rerun()

@st.fragment(run_every=0.5)
def live_monitor():
    st.header("📺 Live Detection Output")
    with job["lock"]:
        frame = job["frame"]
        status = job["status"]
        running = job["running"]
        completed = job["completed"]
        frame_number = job["frame_number"]
        total_detections = job["total_detections"]
        unique_objects = job["unique_objects"]
        class_counts = dict(job["class_counts"])
        detections = list(job["last_detections"])
        output_path = job["output_path"]
        error = job["error"]

    st.info(f"Status: {status}")
    c1, c2, c3 = st.columns(3)
    c1.metric("Processed frames", frame_number)
    c2.metric("Detections across frames", total_detections)
    c3.metric("Unique tracking IDs", unique_objects)

    if frame is not None:
        st.image(frame, caption="Processed video frame", use_container_width=True)
    else:
        st.caption("The processed video preview will appear here after starting.")

    if running:
        if st.button("⏹ Stop Detection", key="stop_detection", use_container_width=True):
            job["stop_event"].set()
            with job["lock"]:
                job["status"] = "Stopping..."
            st.rerun()

    if detections:
        st.subheader("Latest detected objects")
        st.dataframe(pd.DataFrame(detections), use_container_width=True, hide_index=True)

    if class_counts:
        st.subheader("Detection summary")
        summary = pd.DataFrame(
            [{"Object": name, "Detections across frames": count}
             for name, count in sorted(class_counts.items())]
        )
        st.dataframe(summary, use_container_width=True, hide_index=True)
        st.bar_chart(summary.set_index("Object"))

    if completed:
        if error:
            st.error(error)
        elif output_path and Path(output_path).exists():
            st.success("Processing finished. Your video is saved.")
            with open(output_path, "rb") as f:
                st.download_button(
                    "📥 Download processed video", f,
                    file_name="visiontrack_output.mp4",
                    mime="video/mp4", use_container_width=True,
                    key="download_processed_video"
                )

live_monitor()

st.divider()
st.markdown(
    '<p style="text-align:center;color:#a9bdd8;">VisionTrack AI | CodeAlpha Internship Task 4</p>',
    unsafe_allow_html=True
)
