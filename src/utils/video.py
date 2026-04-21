import cv2

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
        
        # Draw background rectangle for text
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - text_height - 5), (x1 + text_width, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return frame
