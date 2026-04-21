import argparse
import cv2
import sys
import os
import yaml

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import YOLODetector
from src.tracking.tracker import ObjectTracker
from src.counting.counter import ZoneCounter
from src.alerts.anomaly import AnomalyDetector
from src.utils.video import VideoStream, draw_detections, draw_zones_and_count
from src.utils.logger import logger

def load_config(config_path):
    """Load YAML configuration file."""
    try:
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    except Exception as e:
        logger.error(f"Failed to load config file {config_path}: {e}")
        sys.exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="AI-powered Smart Inventory Surveillance System")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    return parser.parse_args()

def main():
    args = parse_args()
    
    logger.info("Initializing Smart Inventory Surveillance System...")
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract config values
    model_path = config['model']['path']
    conf_threshold = config['model']['confidence_threshold']
    allowed_classes = config['classes']['allowed_classes']
    class_names = config['classes']['names']
    video_source = config['video']['source']
    display_video = config['video']['display']
    
    # Tracking parameters
    track_max_age = config.get('tracking', {}).get('max_age', 30)
    track_n_init = config.get('tracking', {}).get('n_init', 3)
    track_nn_budget = config.get('tracking', {}).get('nn_budget', 100)
    
    # Counting parameters
    zones_config = config.get('zones', {})
    counting_class = config.get('counting', {}).get('target_class', 'Person')
    db_path = config.get('database', {}).get('path', 'data/inventory.db')
    
    # Alert parameters
    missing_tolerance = config.get('alerts', {}).get('missing_frame_tolerance', 60)
    
    # Initialize components using configuration
    detector = YOLODetector(
        model_path=model_path,
        target_classes=allowed_classes,
        class_names=class_names,
        conf_threshold=conf_threshold
    )
    
    tracker = ObjectTracker(
        max_age=track_max_age,
        n_init=track_n_init,
        nn_budget=track_nn_budget
    )
    
    counter = ZoneCounter(
        zones_config=zones_config,
        target_class=counting_class,
        db_path=db_path
    )
    
    anomaly_engine = AnomalyDetector(
        target_class=counting_class,
        missing_frame_tolerance=missing_tolerance
    )
    
    logger.info(f"Opening video source: {video_source}")
    try:
        video_stream = VideoStream(source=video_source)
    except ValueError as e:
        logger.error(f"Failed to open video source: {e}")
        return

    logger.info("System active. Processing logic started.")
    
    frame_count = 0
    while True:
        ret, frame = video_stream.read()
        if not ret:
            logger.info("End of video stream or cannot read frame.")
            break
            
        frame_count += 1
        
        # 1. Run detection
        detections = detector.detect(frame)
        
        # 2. Run tracking
        tracked_detections = tracker.update(frame, detections)
        
        # 3. Run zone logic
        current_count = counter.update(tracked_detections)
        
        # 4. Run Threat Simulation / Anomaly Logic
        anomaly_engine.evaluate(frame_count, tracked_detections, counter.object_states)
        
        # Log frame activity
        if frame_count % 30 == 0:
            logger.info(f"Frame {frame_count} | Detections: {len(detections)} | Trk: {len(tracked_detections)} | {counting_class}s Counted: {current_count}")
        
        if display_video:
            # 5. Render visually using tracked outputs
            annotated_frame = draw_detections(frame, tracked_detections)
            # Render zones and counter
            annotated_frame = draw_zones_and_count(annotated_frame, zones_config, current_count, counting_class)
            
            cv2.imshow("Inventory Surveillance", annotated_frame)
            
            # Application exit request
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("Termination requested by user.")
                break

    # Clean shutdown
    logger.info("Shutting down resources...")
    video_stream.release()
    cv2.destroyAllWindows()
    logger.info("System shutdown complete.")

if __name__ == "__main__":
    main()
