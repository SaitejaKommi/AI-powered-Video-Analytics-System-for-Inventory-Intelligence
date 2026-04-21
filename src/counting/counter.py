import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger
from src.utils.database import InventoryDatabase

class ZoneCounter:
    def __init__(self, zones_config, target_class, db_path="data/inventory.db"):
        """
        Stateful logic tracking objects across predefined polygon zones.
        
        Args:
            zones_config (dict): Dictionary defining 'storage' and 'exit' paths.
            target_class (str): Only increments count for this specific target label.
            db_path (str): Location of local inventory ledger.
        """
        logger.info(f"Initializing ZoneCounter for target: {target_class}")
        
        # System State Params
        self.target_class = target_class
        self.storage_zone = np.array(zones_config.get('storage', []), np.int32)
        self.exit_zone = np.array(zones_config.get('exit', []), np.int32)
        
        # Active SQLite hook
        self.db = InventoryDatabase(db_path=db_path)
        
        # State tracker: object_id -> "Storage", "Exit", "Outside"
        self.object_states = {}
        
        # Immediately retrieve valid crash-protected counter count 
        self.current_count = self.db.get_current_count()
        logger.info(f"Successfully loaded preserved historical count total: {self.current_count}")

    def _get_bottom_center(self, bbox):
        """Extract the exact bottom center (feet) to check overlap properly vs polygon mapping."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        return (cx, int(y2))

    def _determine_zone(self, point):
        """Identify which exact polygon encompasses the vector center."""
        if cv2.pointPolygonTest(self.storage_zone, point, False) >= 0:
            return "Storage"
        if cv2.pointPolygonTest(self.exit_zone, point, False) >= 0:
            return "Exit"
        return "Outside"

    def update(self, tracked_detections):
        """
        Process the latest tracked inputs to mutate count history and generate DB events cleanly.
        Returns just the current aggregate summation.
        """
        for det in tracked_detections:
            if det["label"].lower() != self.target_class.lower() or det["id"] is None:
                continue
                
            obj_id = det["id"]
            point = self._get_bottom_center(det["bbox"])
            current_zone = self._determine_zone(point)
            
            # Default state memory for untracked inputs is logically outside boundary areas
            previous_zone = self.object_states.get(obj_id, "Outside")
            
            # Only trigger calculation flows upon physical boundary violations
            if previous_zone != current_zone:
                
                # Boundary Check 1: Enter Storage (Meaning incoming stock logic)
                if current_zone == "Storage":
                    self.current_count += 1
                    logger.info(f"[{det['label']} ID:{obj_id}] Entered Storage -> System Count: {self.current_count}")
                    self.db.insert_event(obj_id, "IN", self.current_count)
                
                # Boundary Check 2: Pass specifically from Storage to Exit mapping (Outgoing stock logic)
                elif previous_zone == "Storage" and current_zone == "Exit":
                    self.current_count -= 1
                    logger.info(f"[{det['label']} ID:{obj_id}] Emptied Storage to Exit -> System Count: {self.current_count}")
                    self.db.insert_event(obj_id, "OUT", self.current_count)
                    
            # Memory state assignment preventing duplications inherently tracking IDs indefinitely
            self.object_states[obj_id] = current_zone

        return self.current_count
