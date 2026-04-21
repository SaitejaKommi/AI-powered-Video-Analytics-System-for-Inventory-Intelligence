import sys
import os
from deep_sort_realtime.deepsort_tracker import DeepSort

# Ensure src module access
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

class ObjectTracker:
    def __init__(self, max_age=30, n_init=3, nn_budget=100):
        """
        Initialize the DeepSORT wrapper.
        
        Args:
           max_age (int): Maximum number of missed frames before deleting track.
           n_init (int): Consecutive detections needed to confirm track.
           nn_budget (int): Maximum size of appearance gallery.
        """
        logger.info(f"Initializing DeepSORT Tracker (max_age={max_age}, n_init={n_init})...")
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            nn_budget=nn_budget,
            embedder="mobilenet" # Standard fast embedder
        )
        logger.info("Tracker initialized successfully.")

    def update(self, frame, detections):
        """
        Update the tracker with the latest frame detections.
        
        Args:
            frame (numpy.ndarray): The current video frame.
            detections (list): List of dicts [{"id": None, "label": str, "bbox": [x1,y1,x2,y2], "confidence": float}]
            
        Returns:
            list: Detections updated with persistent IDs ensuring the output format remains exactly as standardized.
        """
        if len(detections) == 0:
            self.tracker.update_tracks([], frame=frame)
            return []

        # Convert structured detections to DeepSORT expected format: ([left, top, w, h], confidence, detection_class)
        bbs = []
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            conf = det["confidence"]
            label = det["label"]
            bbs.append(([x1, y1, w, h], conf, label))
            
        # Update deep-sort tracks
        tracks = self.tracker.update_tracks(bbs, frame=frame)
        
        tracked_detections = []
        for track in tracks:
            # Only return confirmed tracks to avoid flickering on new uncertain objects
            if not track.is_confirmed():
                continue
                
            track_id = int(track.track_id)
            ltrb = track.to_ltrb() # Extract updated [Left, Top, Right, Bottom]
            x1, y1, x2, y2 = [int(v) for v in ltrb]
            
            # DeepSORT stores the label we passed it earlier
            label = track.get_det_class()
            if label is None:
                label = "Unknown"
            
            # Retrieve confidence of the original bounding box tracking hit
            conf = track.get_det_conf()
            if conf is None:
                conf = 0.99
                
            tracked_detections.append({
                "id": track_id,
                "label": label,
                "bbox": [x1, y1, x2, y2],
                "confidence": float(conf)
            })
            
        return tracked_detections
