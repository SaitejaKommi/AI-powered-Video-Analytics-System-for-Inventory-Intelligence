import cv2
import sys
import os
import yaml

# Dynamically link the existing architecture exactly as asked (no duplicating code)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.detection.detector import YOLODetector
from src.tracking.tracker import ObjectTracker
from src.counting.counter import LineCrossingCounter
from src.utils.video import VideoStream, draw_detections, draw_line_and_count

def load_config():
    with open("configs/config.yaml", 'r') as file:
        return yaml.safe_load(file)

def run_debug_validation():
    print("\n==================================")
    print("🚀 INITIATING COUNTING DIAGNOSTICS")
    print("==================================\n")
    
    config = load_config()
    
    # 1. VERIFY VIDEO INPUT
    video_source = "data/samples/test_video.mp4"
    if not os.path.exists(video_source):
        print(f"❌ ERROR: Video file not found at {video_source}")
        return
        
    print(f"✅ Video source confirmed: {video_source}")
    stream = VideoStream(source=video_source)
    
    # Initialize Core Pipelines strictly from architecture configuration
    detector = YOLODetector(
        model_path=config['model']['path'],
        target_classes=config['classes']['allowed_classes'],
        class_names=config['classes']['names'],
        conf_threshold=config['model']['confidence_threshold']
    )
    
    tracker = ObjectTracker(
        track_thresh=config['tracking']['track_thresh'],
        track_buffer=config['tracking']['track_buffer']
    )
    
    counter = LineCrossingCounter(
        vector_config=config['line_crossing']['vector'],
        target_class="Person",
        db_path=":memory:" # Use in-memory SQLite to prevent polluting actual logs
    )
    
    print("\n✅ Pipelines Mounted. Executing Frame Loop...\n")

    frame_skip = 3 # Process 1 frame, skip 3 to speed up CPU inference
    frame_idx = 0
    
    while True:
        ret, frame = stream.read()
        if not ret:
            print("\n✅ End of Video Source gracefully reached.")
            break
            
        frame_idx += 1
        
        # Skip logic to artificially speed up slow CPU environments
        if frame_idx % frame_skip != 0:
            continue
            
        print(f"\n--- [FRAME {frame_idx}] ---")
        
        # 2. DETECTION VALIDATION
        detections = detector.detect(frame)
        print(f"🔍 YOLO Detections: {len(detections)}")
        
        # 3. TRACKING VALIDATION
        tracked_detections = tracker.update(frame, detections)
        active_ids = [d['id'] for d in tracked_detections]
        print(f"🔗 Tracked Targets: {len(tracked_detections)} | IDs: {active_ids}")
        
        # Log exact feet coordinates so user knows where polygons need to be!
        for det in tracked_detections:
            x1, y1, x2, y2 = det["bbox"]
            feet_x = int((x1 + x2) / 2)
            feet_y = int(y2)
            print(f"   📍 Person {det['id']} Feet Coordinate: [X: {feet_x}, Y: {feet_y}]")
            
        # Store internal state pre-update to diff changes clearly
        previous_count = counter.current_count
        previous_states = dict(counter.object_states)
        
        # 4. COUNTING VALIDATION
        current_count = counter.update(tracked_detections)
        
        # Manually extract the transitions explicitly for the console
        for det in tracked_detections:
            obj_id = det["id"]
            if obj_id is None: continue
            
            p_zone = previous_states.get(obj_id, "Outside")
            c_zone = counter.object_states.get(obj_id, "Outside")
            
            if p_zone != c_zone:
                print(f"   ⚠️ ZONE TRANSITION -> Person {obj_id} walked from '{p_zone}' to '{c_zone}'")
                
        if current_count != previous_count:
             print(f"   📈 COUNT UPDATED -> {previous_count} changed to {current_count}")
             
        # 5. DEBUG SUPPORT VISUALIZATION
        annotated_frame = draw_detections(frame, tracked_detections)
        annotated_frame = draw_line_and_count(annotated_frame, config['line_crossing']['vector'], current_count, "Person")
        
        # Visual Debug output
        annotated_frame = cv2.resize(annotated_frame, (800, 600))
        cv2.imshow("Proxy Test Diagnostics (Optimized)", annotated_frame)
        
        # Speed up opencv delay
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n❌ Execution manually terminated.")
            break

    # 6. ERROR HANDLING
    stream.release()
    cv2.destroyAllWindows()
    print("\n✅ Validations complete.\n")

if __name__ == "__main__":
    run_debug_validation()
