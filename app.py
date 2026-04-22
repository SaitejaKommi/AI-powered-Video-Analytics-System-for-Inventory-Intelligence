import streamlit as st
import cv2
import yaml
import sys
import os
import pandas as pd

# Append path ensuring src imports cleanly
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.detection.detector import YOLODetector
from src.tracking.tracker import ObjectTracker
from src.counting.counter import LineCrossingCounter
from src.alerts.anomaly import AnomalyDetector
from src.utils.database import InventoryDatabase
from src.utils.video import VideoStream, draw_detections, draw_line_and_count
from src.utils.logger import logger

st.set_page_config(page_title="Inventory Security Dashboard", layout="wide", page_icon="🏭")

# --- INITIALIZATION ---
def load_config(config_path="configs/config.yaml"):
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config file: {e}")
        return {}

config = load_config()

# Extract parameters
model_path = config.get('model', {}).get('path')
conf_threshold = config.get('model', {}).get('confidence_threshold', 0.5)
allowed_classes = config.get('classes', {}).get('allowed_classes', [0])
class_names = config.get('classes', {}).get('names', {0: 'Person'})
video_source = config.get('video', {}).get('source', 0)

track_thresh = config.get('tracking', {}).get('track_thresh', 0.3)
track_buffer = config.get('tracking', {}).get('track_buffer', 60)
match_thresh = config.get('tracking', {}).get('match_thresh', 0.8)

line_config = config.get('line_crossing', {}).get('vector', [960, 0, 960, 2000])
counting_class = config.get('counting', {}).get('target_class', 'Person')
db_path = config.get('database', {}).get('path', 'data/inventory.db')

missing_tolerance = config.get('alerts', {}).get('missing_frame_tolerance', 60)

# Instantiate Database globally
db = InventoryDatabase(db_path=db_path)

@st.cache_resource
def init_ml_pipeline():
    logger.info("Initializing heavy CV ML models...")
    detector = YOLODetector(model_path, allowed_classes, class_names, conf_threshold)
    tracker = ObjectTracker(track_thresh=track_thresh, track_buffer=track_buffer, match_thresh=match_thresh)
    return detector, tracker

detector, tracker = init_ml_pipeline()

# Instantiate lightweight geometry and logic engines cleanly so they hot-swap when config changes
counter = LineCrossingCounter(line_config, counting_class, db_path)
anomaly_engine = AnomalyDetector(counting_class, missing_tolerance, db_path)

# --- UI LAYOUT ---
st.title("🏭 Smart Inventory Surveillance")

st.sidebar.header("⚙️ Configuration")
st.sidebar.text(f"Model: {model_path}")
st.sidebar.text(f"Target: {counting_class}")
st.sidebar.text(f"Video Source: {video_source}")
st.sidebar.divider()
st.sidebar.subheader("Active Feed Control")
st.sidebar.text(f"Live Line Vector: {line_config}")

if 'stream_active' not in st.session_state:
    st.session_state.stream_active = False

def start_stream():
    st.session_state.stream_active = True

def stop_stream():
    st.session_state.stream_active = False

st.sidebar.button("▶️ Start Server", on_click=start_stream, use_container_width=True)
st.sidebar.button("⏹️ Stop Server", on_click=stop_stream, use_container_width=True)

col1, col2 = st.columns([2.5, 1])

with col1:
    st.subheader("Live Security Feed")
    video_placeholder = st.empty()

with col2:
    st.subheader("📦 Inventory Tracking")
    count_placeholder = st.empty()
    st.caption("Recent Activity Log")
    events_placeholder = st.empty()
    
    st.subheader("🚨 Threat Alerts")
    alerts_placeholder = st.empty()

# --- ORCHESTRATION LOOP ---
if st.session_state.stream_active:
    try:
        video_stream = VideoStream(source=video_source)
        frame_count = 0
        
        while st.session_state.stream_active:
            ret, frame = video_stream.read()
            if not ret:
                st.error("Lost connection to video source.")
                break
                
            frame_count += 1
            
            # --- AI Core Logic ---
            detections = detector.detect(frame)
            tracked_detections = tracker.update(frame, detections)
            current_count = counter.update(tracked_detections)
            anomaly_engine.evaluate(frame_count, tracked_detections, counter.object_states)
            
            # --- UI Updates ---
            if frame_count % 3 == 0:  # Mild throttle for render stability
                # Annotate natively
                frame_annotated = draw_detections(frame, tracked_detections)
                frame_annotated = draw_line_and_count(frame_annotated, line_config, current_count, counting_class)
                
                # Convert BGR to RGB for Streamlit accurately
                frame_rgb = cv2.cvtColor(frame_annotated, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                
                # Update System Count HUD
                count_placeholder.metric(label="Current Stock", value=current_count)
                
                # Update System Tracking Ledger
                recent_events = db.get_recent_events(limit=5)
                if recent_events:
                    events_df = pd.DataFrame(recent_events, columns=["Time", "Action", "ID Tracker", "Stock Remaining"])
                    events_placeholder.dataframe(events_df, hide_index=True, use_container_width=True)
                else:
                    events_placeholder.info("Awaiting structural stock movement.")
                
                # Update Threat Alert Table
                recent_alerts = db.get_recent_alerts(limit=5)
                if recent_alerts:
                    alert_df = pd.DataFrame(recent_alerts, columns=["Timestamp", "Type", "ID target", "Drop Zone"])
                    # Use styling to make errors pop
                    alerts_placeholder.dataframe(alert_df, hide_index=True, use_container_width=True)
                else:
                    alerts_placeholder.success("No anomalies detected.")
                
    except Exception as e:
        st.error(f"Camera Initialization cleanly aborted: {e}")
    finally:
        if 'video_stream' in locals():
            video_stream.release()
else:
    video_placeholder.info("Stream offline. Click 'Start Server' in the sidebar.")
    
    # Still show passive state if offline
    current_count = db.get_current_count()
    count_placeholder.metric(label="Current Stock (Offline)", value=current_count)
    
    recent_events = db.get_recent_events(limit=5)
    if recent_events:
        events_df = pd.DataFrame(recent_events, columns=["Time", "Action", "ID Tracker", "Stock Remaining"])
        events_placeholder.dataframe(events_df, hide_index=True, use_container_width=True)
    else:
        events_placeholder.info("Awaiting structural stock movement.")
    recent_alerts = db.get_recent_alerts(limit=5)
    if recent_alerts:
        alert_df = pd.DataFrame(recent_alerts, columns=["Timestamp", "Type", "ID target", "Drop Zone"])
        alerts_placeholder.dataframe(alert_df, hide_index=True, use_container_width=True)
    else:
        alerts_placeholder.success("No anomalies historically logged.")
