import sys
import os
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

class ZoneCounter:
    def __init__(self, zones_config, target_class):
        """
        Stateful logic tracking objects across predefined polygon zones.
        
        Args:
            zones_config (dict): Dictionary defining 'storage' and 'exit' coordinates.
            target_class (str): Only increments count for this specific class label.
        """
        logger.info(f"Initializing ZoneCounter for target: {target_class}")
        self.target_class = target_class
        self.storage_zone = np.array(zones_config.get('storage', []), np.int32)
        self.exit_zone = np.array(zones_config.get('exit', []), np.int32)
        
        # State tracker: object_id -> "Storage", "Exit", "Outside"
        self.object_states = {}
        self.current_count = 0

    def _get_bottom_center(self, bbox):
        """Extract the bottom center point of the bounding box mapping to object's feet."""
        x1, y1, x2, y2 = bbox
        cx = int((x1 + x2) / 2)
        return (cx, int(y2))

    def _determine_zone(self, point):
        """Return the zone name a 2D point resides inside."""
        if cv2.pointPolygonTest(self.storage_zone, point, False) >= 0:
            return "Storage"
        if cv2.pointPolygonTest(self.exit_zone, point, False) >= 0:
            return "Exit"
        return "Outside"

    def update(self, tracked_detections):
        """
        Process the latest detections to track movement patterns and evaluate increments/decrements.
        Returns the identical structured list so pipeline flows unchanged, but internal states mutate.
        """
        for det in tracked_detections:
            # We strictly count targets configured by user
            if det["label"].lower() != self.target_class.lower() or det["id"] is None:
                continue
                
            obj_id = det["id"]
            point = self._get_bottom_center(det["bbox"])
            current_zone = self._determine_zone(point)
            
            # Extract historical location
            previous_zone = self.object_states.get(obj_id, "Outside")
            
            if previous_zone != current_zone:
                # Rule 1: Enters Storage from anywhere else (typically Outside or Exit)
                if current_zone == "Storage":
                    self.current_count += 1
                    logger.info(f"[{det['label']} ID:{obj_id}] Entered Storage -> Count: {self.current_count}")
                
                # Rule 2: Leaves Storage -> enters Exit
                elif previous_zone == "Storage" and current_zone == "Exit":
                    self.current_count -= 1
                    logger.info(f"[{det['label']} ID:{obj_id}] Moved Storage to Exit -> Count: {self.current_count}")
                    
            # Persist local state boundary memory
            self.object_states[obj_id] = current_zone

        return self.current_count
