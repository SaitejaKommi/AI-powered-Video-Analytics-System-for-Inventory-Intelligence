# AI-powered Smart Inventory Surveillance System

A production-ready computer vision solution for detecting and tracking cement bags and personnel, specifically designed for cement shop environments.

## Features

- **Real-time Object Detection:** Leverages YOLOv8 to detect humans (and eventually cement bags).
- **Custom Bounding Boxes:** Clean and optimized bounding box drawing using OpenCV.
- **Modular Architecture:** Well-organized source code separating detection, tracking, counting, and logic.

## Project Structure

```text
smart-inventory-ai/
│
├── data/
│   ├── videos/       # Store full-length videos here
│   └── samples/      # Store short sample clips
│
├── models/
│   └── yolov8/       # Place your custom yolov8 models here (e.g. best.pt)
│
├── src/
│   ├── detection/    # YOLOv8 loading and inference logic
│   ├── tracking/     # DeepSORT integration (coming soon)
│   ├── counting/     # Zone entrance/exit logic (coming soon)
│   ├── alerts/       # Anomaly alerts (coming soon)
│   ├── utils/        # Video reading & drawing utils
│   └── main.py       # Main entry point pipeline
│
├── configs/          # Configuration files
├── outputs/          # Output videos and logs
├── requirements.txt  # Dependencies
├── .gitignore        # Git ignores
└── README.md         # Project documentation
```

## Setup Instructions

1. **Create a virtual environment (Recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

To run the real-time detection pipeline, use the `main.py` script:

**Using WebCam:**
```bash
python src/main.py --source 0
```

**Using a Sample Video:**
```bash
python src/main.py --source data/samples/test_video.mp4
```

Press **`q`** to quit the video stream.
