import sqlite3
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

class InventoryDatabase:
    def __init__(self, db_path="data/inventory.db"):
        """
        Initialize connection to SQLite tracking database.
        """
        self.db_path = db_path
        
        # Ensure parent directory physically exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self.init_db()

    def init_db(self):
        """Create the schema securely if it does not already exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Enforcing atomic transactions with sqlite context contexts
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS inventory_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        object_id INTEGER,
                        event_type TEXT,
                        count_after_event INTEGER
                    )
                ''')
                conn.commit()
                logger.info(f"Database schema initialized accurately at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")

    def insert_event(self, object_id, event_type, count_after_event):
        """
        Perform an atomic transaction to append an event directly to disk.
        
        Args:
            object_id (int): DeepSORT tracking ID triggering the boundary cross.
            event_type (str): 'IN' (arrival to storage) or 'OUT' (exit from storage).
            count_after_event (int): The current truth aggregate tally.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute(
                    '''INSERT INTO inventory_events (timestamp, object_id, event_type, count_after_event) 
                       VALUES (?, ?, ?, ?)''',
                    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), object_id, event_type, count_after_event)
                )
                conn.commit()
        except Exception as e:
            logger.error(f"DB Write Failure - Could not log '{event_type}' for Object ID '{object_id}': {e}")

    def get_current_count(self):
        """
        Fetch the active aggregate count by checking the mathematically accurate terminal entry on the ledger.
        Defaults securely to 0.
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT count_after_event FROM inventory_events ORDER BY id DESC LIMIT 1')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to cleanly resurrect count from DB. Defaulting to 0: {e}")
            return 0
