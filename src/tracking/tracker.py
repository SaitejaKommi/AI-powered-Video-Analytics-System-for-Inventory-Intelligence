import sys
import os
import numpy as np
import supervision as sv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

class ObjectTracker:
    def __init__(self, track_thresh=0.3, track_buffer=60, match_thresh=0.8):
        """
        Initialize the ByteTrack engine via Supervision.
        """
        logger.info(f"Initializing ByteTrack Engine (thresh={track_thresh}, patience={track_buffer}f)...")
        self.tracker = sv.ByteTrack(
            track_activation_threshold=track_thresh,
            lost_track_buffer=track_buffer,
            minimum_matching_threshold=match_thresh
        )
        # Hardcoding the map strictly for standardizing the dictionary relay back to `counter.py`
        self.class_mapping = {"Person": 0, "Cement Bag": 1}
        self.rev_mapping = {0: "Person", 1: "Cement Bag"}
        
        logger.info("Tracker initialized successfully.")

    def update(self, frame, detections):
        """
        Update the tracker using Supervision's rapid math arrays.
        """
        if len(detections) == 0:
            return []

        # 1. Translate structural dicts to sv.Detections numpy blocks
        xyxy = np.array([d["bbox"] for d in detections])
        confidence = np.array([d["confidence"] for d in detections])
        class_ids = np.array([self.class_mapping.get(d["label"], 0) for d in detections])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_ids
        )
        
        # 2. Run ByteTrack core IoU engine
        tracked = self.tracker.update_with_detections(sv_detections)
        
        # 3. Restructure back to JSON-style dicts to protect modular interoperability
        tracked_detections = []
        for i in range(len(tracked.xyxy)):
            tid = tracked.tracker_id[i]
            if tid is None:
                continue
                
            x1, y1, x2, y2 = tracked.xyxy[i].astype(int)
            conf = float(tracked.confidence[i]) if tracked.confidence is not None else 1.0
            label = self.rev_mapping.get(int(tracked.class_id[i]), "Unknown")
            
            tracked_detections.append({
                "id": int(tid),
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": conf
            })
            
        return tracked_detections
