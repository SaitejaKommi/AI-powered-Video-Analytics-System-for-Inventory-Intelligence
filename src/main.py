import argparse
import cv2
import sys
import os
import yaml

# Add project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import YOLODetector
from src.utils.video import VideoStream, draw_detections
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
    
    # Initialize components using configuration
    detector = YOLODetector(
        model_path=model_path,
        target_classes=allowed_classes,
        class_names=class_names,
        conf_threshold=conf_threshold
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
        
        # Run detection mapping cleanly to pure structured data
        detections = detector.detect(frame)
        
        # Log frame activity
        if frame_count % 30 == 0:
            # Throttling the log slightly so it's readable, but logging detection count
            logger.info(f"Frame {frame_count} processed - Detections: {len(detections)}")
        
        if display_video:
            # Render visually
            annotated_frame = draw_detections(frame, detections)
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
