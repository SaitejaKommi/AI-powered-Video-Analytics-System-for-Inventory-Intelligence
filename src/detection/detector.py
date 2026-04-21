from ultralytics import YOLO
import numpy as np
from src.utils.logger import logger

class YOLODetector:
    def __init__(self, model_path, target_classes=None, class_names=None, conf_threshold=0.5):
        """
        Initialize the YOLOv8 detector using configuration parameters.
        
        Args:
            model_path (str): Path to the YOLOv8 weights file.
            target_classes (list): List of class IDs to detect. If None, detects all.
            class_names (dict): Mapping of class ID to string label.
            conf_threshold (float): Minimum confidence threshold.
        """
        self.model_path = model_path
        logger.info(f"Loading YOLOv8 model from {model_path}...")
        self.model = YOLO(model_path)
        logger.info("Model loaded successfully.")
        
        self.target_classes = target_classes
        self.class_names = class_names if class_names else {}
        self.conf_threshold = conf_threshold

    def detect(self, frame):
        """
        Perform object detection on a single frame and return tracking-ready structured data.

        Args:
            frame (numpy.ndarray): The input video frame.

        Returns:
            list: List of dictionaries containing structured detection data.
        """
        results = self.model(frame, classes=self.target_classes, verbose=False)
        
        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                conf = float(box.conf[0].cpu().numpy())
                
                # Apply confidence threshold
                if conf < self.conf_threshold:
                    continue
                    
                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Class mapping
                cls_id = int(box.cls[0].cpu().numpy())
                label = self.class_names.get(cls_id, f"Class_{cls_id}")
                
                # Tracking-ready output format
                detection = {
                    "id": None,                 # Placeholder for DeepSORT later
                    "label": label,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": conf
                }
                
                detections.append(detection)
                
        return detections
