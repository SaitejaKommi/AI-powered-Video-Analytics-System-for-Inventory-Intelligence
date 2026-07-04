# 🏭 AI-Powered Video Analytics System for Inventory Intelligence

A production-grade, highly resilient computer vision solution designed to automatically track inventory (cement bags) and personnel. Optimized for local edge deployments (e.g., shop-floor laptops) monitoring live CCTV streams in industrial environments.

---

## 🚀 Key Features

* **Real-Time Edge Detection & Tracking**: Combines **YOLOv8** object detection with **ByteTrack** multi-object tracking to persistently track moving targets across frames.
* **Asynchronous Database Pipeline**: Employs a background daemon worker thread and queueing system (`queue.Queue`) for SQLite updates. This prevents disk I/O operations from blocking the main frame-processing loop.
* **Database Resilience (WAL Mode)**: Configured with SQLite Write-Ahead Logging (`WAL`) and synchronous normal mode, eliminating database locks during concurrent reads/writes and protecting against corruption from power cuts.
* **Fault-Tolerant Camera Auto-Reconnect**: Implements automated video stream recovery. If the CCTV or USB feed drops, the system sleeps and retries connecting up to 10 times without crashing the execution loop.
* **UI Rate Decoupling**: Decouples the Streamlit UI refresh rate (~4 FPS) from the backend AI inference engine (30+ FPS), significantly lowering CPU usage and preventing thermal throttling on edge devices.
* **Robust Configuration Validation**: Features a fail-fast YAML validation engine that inspects config files at boot to protect the pipeline from invalid geometry vectors, float boundaries, or typos.

---

## 📁 Directory Structure

```text
smart-inventory-analytics/
├── configs/
│   └── config.yaml          # System configuration file (thresholds, coordinates, classes)
├── data/
│   ├── inventory.db         # SQLite persistent database (automatically created)
│   └── videos/              # Video storage folder
├── models/
│   └── yolov8/              # Place trained models here (e.g., yolov8n.pt)
├── src/
│   ├── alerts/              # Anomaly detection & email notification engine
│   ├── counting/            # Line-crossing mathematical counting logic
│   ├── detection/           # YOLOv8 class wrapper and inference orchestration
│   ├── tracking/            # ByteTrack object tracking persistence
│   └── utils/               # SQLite database client, logger, and video utilities
├── scripts/
│   └── validate_counting.py # Headless diagnostics utility for pipeline regression checks
├── app.py                   # Streamlit production dashboard entry point
├── requirements.txt         # Project dependencies
└── README.md                # System documentation
```

---

## ⚙️ Setup & Installation

### 1. Prerequisite Environments
Make sure Python 3.8+ and `pip` are installed on your machine.

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv

# Activate on Linux/macOS:
source venv/bin/activate

# Activate on Windows (PowerShell):
venv\Scripts\Activate.ps1
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🛠️ Configuration (`configs/config.yaml`)

Manage the application behaviour dynamically via `configs/config.yaml`:

```yaml
model:
  path: "models/yolov8/yolov8n.pt"  # Path to YOLO weights
  confidence_threshold: 0.5         # Minimum confidence for detection

classes:
  allowed_classes: [0, 1]           # COCO IDs (0: Person, 1: Cement Bag proxy)

video:
  source: 0                         # CCTV stream url, video file path, or webcam ID

tracking:
  track_thresh: 0.3                 # ByteTrack detection association threshold
  track_buffer: 60                  # Frame window to retain lost object IDs

line_crossing:
  vector: [960, 0, 960, 2000]       # Tripwire coordinates [x1, y1, x2, y2]
```

---

## 🖥️ Running the System

### 1. Launch the Streamlit Dashboard (Production)
The interactive dashboard handles live rendering, metrics overlays, logging ledgers, and database query streams:
```bash
streamlit run app.py
```

### 2. Headless Diagnostic Validation
Validate the detection, tracking, database queue, and line-crossing logic without launching a browser UI:
```bash
python scripts/validate_counting.py
```

---

## 🛡️ Phase 1 Validation & Stability Status

The infrastructure modifications implemented in Phase 1 have been audited to ensure robust operation on local shop floor machines. For detailed verification logs and audit records, see:
* [Phase 1 Validation Report](file:///C:/Users/kommi/.gemini/antigravity-ide/brain/21bbc0ad-9c1d-4fec-b344-2cff85482d64/phase1_validation_report.md)
* [Phase 1 Readiness Checklist](file:///C:/Users/kommi/.gemini/antigravity-ide/brain/21bbc0ad-9c1d-4fec-b344-2cff85482d64/phase1_readiness_checklist.md)
