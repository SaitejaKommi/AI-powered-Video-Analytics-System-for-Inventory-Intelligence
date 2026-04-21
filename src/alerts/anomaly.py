import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

class AnomalyDetector:
    def __init__(self, target_class, missing_frame_tolerance=60):
        """
        Engine evaluating temporal behavior flows to alert on suspect drops.
        
        Args:
            target_class (str): Evaluate rules specifically on this tracked string (e.g. "Person" or "Cement Bag")
            missing_frame_tolerance (int): Grace-limit of frames before a vanished un-exited object trips bells.
        """
        logger.info(f"Initializing AnomalyDetector Engine (Allowance: {missing_frame_tolerance} Frames)")
        self.target_class = target_class
        self.missing_frame_tolerance = missing_frame_tolerance
        
        # Memory states structured safely: { "id": {"last_zone": "Storage", "last_frame": 120} }
        self.object_history = {}
        
        # Safegaurd ensuring we don't spam 100 alerts a second for the same incident
        self.alerted_ids = set()

    def evaluate(self, current_frame, current_detections, zone_states):
        """
        Analyze current frame vectors vs historical memory arrays for logical theft rule-breaches.
        
        Args:
            current_frame (int): Monotonically increasing index from main iteration counter.
            current_detections (list): Actively seen tracking dicts right now.
            zone_states (dict): Snapshot dict from ZoneCounter telling us exactly where things officially stood.
        """
        active_ids_in_frame = set()
        
        # 1. Update Historical Memory 
        for det in current_detections:
            if det['label'].lower() != self.target_class.lower() or det['id'] is None:
                continue
                
            obj_id = det['id']
            active_ids_in_frame.add(obj_id)
            
            # Record last frame ping and exactly where the count logically believed it to be
            self.object_history[obj_id] = {
                "last_zone": zone_states.get(obj_id, "Outside"),
                "last_frame": current_frame
            }

        # 2. Sweep Memory looking for Rule Breakers
        for obj_id, history in list(self.object_history.items()):
            # Rule out objects we've already screamed about
            if obj_id in self.alerted_ids:
                continue
                
            # If we didn't see the object THIS frame, investigate.
            if obj_id not in active_ids_in_frame:
                frame_gap = current_frame - history["last_frame"]
                
                # Did it exceed tolerance limit?
                if frame_gap > self.missing_frame_tolerance:
                    
                    # Rule 1 Violation: Did it vanish natively inside secure premises?
                    if history["last_zone"] == "Storage":
                        self.trigger_alert("EVAPORATION_ANOMALY", obj_id, history)
                        self.alerted_ids.add(obj_id)
                        
                    # Standard housekeeping: Clean up memory so we aren't leaking resources looping over stale valid exits
                    elif history["last_zone"] == "Exit" or history["last_zone"] == "Outside":
                         del self.object_history[obj_id]

    def trigger_alert(self, alert_type, obj_id, history_snapshot):
        """Standardized router for system warnings."""
        logger.error("=========================================")
        logger.error(f"🚨 SECURITY ALERT: {alert_type} 🚨")
        logger.error(f"Target: {self.target_class} [ID: {obj_id}]")
        logger.error(f"Dropped From Camera Completely Inside: {history_snapshot['last_zone']}")
        logger.error(f"Last Known Contact Ping @ Frame: {history_snapshot['last_frame']}")
        logger.error("=========================================")
