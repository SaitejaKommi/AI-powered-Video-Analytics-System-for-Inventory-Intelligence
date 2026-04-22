import cv2
import numpy as np
import supervision as sv

class VideoStream:
    def __init__(self, source=0):
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()

def draw_line_and_count(frame, vector_config, count, target_class):
    """
    Draw mathematical Tripwire and the active counter overlay on the frame natively using `sv.LineZoneAnnotator`.
    """
    if not vector_config or len(vector_config) != 4:
        return frame
        
    start = sv.Point(vector_config[0], vector_config[1])
    end = sv.Point(vector_config[2], vector_config[3])
    
    # 1. Overlay colored line safely using native cv2 fallback instead of invoking heavy generic UI rendering
    # LineZoneAnnotator expects a LineZone object entirely. Since we are just drawing a visual vector cleanly:
    cv2.line(frame, (start.x, start.y), (end.x, end.y), (0, 165, 255), 4)
    
    # Add floating designator natively centered on line
    cx = int((start.x + end.x) / 2)
    cy = int((start.y + end.y) / 2)
    cv2.putText(frame, "Tripwire", (cx + 10, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
    # 2. Draw Counter Status overlay
    count_text = f"{target_class} Count: {count}"
    
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
    
    cv2.putText(frame, count_text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame

def draw_detections(frame, detections):
    """
    Draw bounding boxes, labels, and tracking IDs on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        track_id = det["id"]
        
        id_text = f" ID:{track_id}" if track_id is not None else ""
        text = f"{label}{id_text}: {conf:.2f}"
        
        color = (0, 255, 0) if label.lower() == "person" else (0, 165, 255) 
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Center of bounding box is the anchor used by ByteTrack geometrically
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame
