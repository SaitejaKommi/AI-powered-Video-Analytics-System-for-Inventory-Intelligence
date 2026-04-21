import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger
from src.utils.database import InventoryDatabase
from src.alerts.notifier import AlertManager
from datetime import datetime

class AnomalyDetector:
    def __init__(self, target_class, missing_frame_tolerance=60, db_path="data/inventory.db"):
        """
        Engine evaluating temporal behavior flows to alert on suspect drops.
        
        Args:
            target_class (str): Evaluate rules specifically on this target label.
            missing_frame_tolerance (int): Grace limit of frames before vanished object trips alarms.
            db_path (str): DB path to commit alerts logs natively.
        """
        logger.info(f"Initializing AnomalyDetector Engine (Allowance: {missing_frame_tolerance} Frames)")
        self.target_class = target_class
        self.missing_frame_tolerance = missing_frame_tolerance
        
        self.db = InventoryDatabase(db_path=db_path)
        self.notifier = AlertManager()
        
        # Memory states structured safely: { "id": {"last_zone": "Storage", "last_frame": 120} }
        self.object_history = {}
        
        # Guard ensuring we don't spam duplicate alerts for the same ID
        self.alerted_ids = set()

    def evaluate(self, current_frame, current_detections, zone_states):
        """
        Analyze current frame vectors vs historical memory arrays for logical theft rule-breaches.
        """
        active_ids_in_frame = set()
        
        # 1. Update Historical Memory 
        for det in current_detections:
            if det['label'].lower() != self.target_class.lower() or det['id'] is None:
                continue
                
            obj_id = det['id']
            active_ids_in_frame.add(obj_id)
            
            # Record last frame ping
            self.object_history[obj_id] = {
                "last_zone": zone_states.get(obj_id, "Outside"),
                "last_frame": current_frame
            }

        # 2. Sweep Memory looking for Event Breaches
        for obj_id, history in list(self.object_history.items()):
            if obj_id in self.alerted_ids:
                continue
                
            # If object is missing THIS frame, investigate
            if obj_id not in active_ids_in_frame:
                frame_gap = current_frame - history["last_frame"]
                
                if frame_gap > self.missing_frame_tolerance:
                    # Rule 1 Violation: Vanished from secure premises natively
                    if history["last_zone"] == "Storage":
                        self.trigger_alert("EVAPORATION_ANOMALY", obj_id, history)
                        self.alerted_ids.add(obj_id)
                        
                    # Housekeeping
                    elif history["last_zone"] in ["Exit", "Outside"]:
                         del self.object_history[obj_id]

    def trigger_alert(self, alert_type, obj_id, history_snapshot):
        """Router for tracking alerts onto disk and terminal logging."""
        last_zone = history_snapshot['last_zone']
        
        logger.error("=========================================")
        logger.error(f"🚨 SECURITY ALERT: {alert_type} 🚨")
        logger.error(f"Target: {self.target_class} [ID: {obj_id}]")
        logger.error(f"Dropped From Camera Completely Inside: {last_zone}")
        logger.error("=========================================")
        
        # Push cleanly to the connected DB architecture
        self.db.insert_alert(alert_type, obj_id, last_zone)
        
        # Fire active push notifications asynchronously
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details = f"Target was explicitly last confirmed inside '{last_zone}' and vanished illegally."
        self.notifier.notify(timestamp, alert_type, obj_id, details)
