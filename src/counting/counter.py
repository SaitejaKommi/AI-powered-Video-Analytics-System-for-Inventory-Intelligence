import sys
import os
import numpy as np
import supervision as sv

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger
from src.utils.database import InventoryDatabase

class LineCrossingCounter:
    def __init__(self, vector_config, target_class, db_path="data/inventory.db"):
        """
        Stateful logic tracking objects across a predefined Tripwire Vector.
        """
        logger.info(f"Initializing LineCrossingCounter for target: {target_class}")
        
        self.target_class = target_class
        
        # Unpack [x1, y1, x2, y2] securely
        if not vector_config or len(vector_config) != 4:
            logger.warning(f"Invalid Line Vector {vector_config}. Defaulting to absolute center.")
            vector_config = [960, 0, 960, 1080]
            
        start = sv.Point(vector_config[0], vector_config[1])
        end = sv.Point(vector_config[2], vector_config[3])
        
        # Instantiate Supervision math engine natively
        self.line_zone = sv.LineZone(start=start, end=end)
        
        # Active SQLite hook
        self.db = InventoryDatabase(db_path=db_path)
        
        # Immediately retrieve valid crash-protected counter count 
        self.current_count = self.db.get_current_count()
        logger.info(f"Successfully loaded preserved historical count total: {self.current_count}")
        
        # State tracker to satisfy AnomalyDetector legacy hooks
        self.inside_objects = set()
        
        # Internal map for tracking names strictly matching target constraints
        self.class_mapping = {"Person": 0, "Cement Bag": 1}

    @property
    def object_states(self):
        """Emulate the zone dictionary interface for the AnomalyDetector without breaking modularity."""
        return {tid: "Storage" for tid in self.inside_objects}

    def update(self, tracked_detections):
        """
        Process the latest tracked inputs to mutate count history and generate DB events cleanly.
        Because Supervision handles the state cache, we query it instantly for crossings.
        """
        if len(tracked_detections) == 0:
            return self.current_count

        # 1. Strip out non-target detections so they don't corrupt the cross count
        filtered_targets = [d for d in tracked_detections if d["label"].lower() == self.target_class.lower()]
        
        if len(filtered_targets) == 0:
            return self.current_count

        # 2. Reshape to sv.Detections
        xyxy = np.array([d["bbox"] for d in filtered_targets])
        tracker_ids = np.array([d["id"] for d in filtered_targets])
        class_ids = np.array([self.class_mapping.get(d["label"], 0) for d in filtered_targets])
        
        sv_detections = sv.Detections(
            xyxy=xyxy,
            tracker_id=tracker_ids,
            class_id=class_ids
        )
        
        # 3. Fire geometric math
        crossed_in, crossed_out = self.line_zone.trigger(detections=sv_detections)
        
        # 4. Map booleans natively back to Tracker IDs for database persistence
        for idx in range(len(sv_detections)):
            tid = int(sv_detections.tracker_id[idx])
            label = self.target_class
            
            if crossed_in[idx]:
                self.inside_objects.add(tid)
                self.current_count += 1
                logger.info(f"[{label} ID:{tid}] Breached Line INWARDS -> System Count: {self.current_count}")
                self.db.insert_event(tid, "IN", self.current_count)
                
            elif crossed_out[idx]:
                self.inside_objects.discard(tid)
                self.current_count -= 1
                logger.info(f"[{label} ID:{tid}] Breached Line OUTWARDS -> System Count: {self.current_count}")
                self.db.insert_event(tid, "OUT", self.current_count)

        return self.current_count
