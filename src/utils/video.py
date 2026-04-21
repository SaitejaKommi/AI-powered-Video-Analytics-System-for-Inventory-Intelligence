import cv2
import numpy as np

class VideoStream:
    def __init__(self, source=0):
        """
        Initialize video capture.
        
        Args:
            source (int/str): Video source. 0 for webcam, or path to video file.
        """
        # Ensure correct type for webcam vs file
        if isinstance(source, str) and source.isdigit():
            source = int(source)
            
        self.source = source
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Unable to open video source: {source}")

    def read(self):
        """
        Read a single frame from the video stream.
        
        Returns:
            tuple: (ret (bool), frame (numpy.ndarray))
        """
        return self.cap.read()

    def release(self):
        """Release the video stream."""
        self.cap.release()


def draw_zones_and_count(frame, zones_config, count, target_class):
    """
    Draw polygon zones and the active counter overlay on the frame.
    """
    # Draw Storage Zone (Blue)
    if 'storage' in zones_config:
        storage_pts = np.array(zones_config['storage'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [storage_pts], isClosed=True, color=(255, 0, 0), thickness=2)
        cv2.putText(frame, "Storage Zone", (storage_pts[0][0][0], storage_pts[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    
    # Draw Exit Zone (Red)
    if 'exit' in zones_config:
        exit_pts = np.array(zones_config['exit'], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [exit_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.putText(frame, "Exit Zone", (exit_pts[0][0][0], exit_pts[0][0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    
    # Draw Counter Status overlay
    count_text = f"{target_class} Count: {count}"
    
    # Background for text
    (tw, th), _ = cv2.getTextSize(count_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
    cv2.rectangle(frame, (10, 10), (20 + tw, 20 + th + 10), (0, 0, 0), -1)
    
    # White text on black background
    cv2.putText(frame, count_text, (15, 15 + th), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return frame


def draw_detections(frame, detections):
    """
    Draw bounding boxes, labels, and tracking IDs on the frame.
    
    Args:
        frame (numpy.ndarray): The image frame.
        detections (list): List of structured detection dictionaries.
                           Format: {"id": int/None, "label": str, "bbox": [x1,y1,x2,y2], "confidence": float}
    
    Returns:
        numpy.ndarray: The annotated frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        label = det["label"]
        conf = det["confidence"]
        track_id = det["id"]
        
        # Display ID if tracking is active
        id_text = f" ID:{track_id}" if track_id is not None else ""
        text = f"{label}{id_text}: {conf:.2f}"
        
        # Determine color based on label (future-proofing for multiple classes)
        color = (0, 255, 0) if label.lower() == "person" else (0, 165, 255) 
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Overlay a dot at the bottom center to show exactly what's being evaluated for counts
        cx = int((x1 + x2) / 2)
        cv2.circle(frame, (cx, y2), 4, (0, 255, 255), -1)
        
        # Draw background rectangle for text
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame
