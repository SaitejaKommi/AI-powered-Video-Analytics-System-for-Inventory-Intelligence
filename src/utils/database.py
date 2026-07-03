import sqlite3
import os
import sys
import threading
import queue
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils.logger import logger

class InventoryDatabase:
    def __init__(self, db_path="data/inventory.db"):
        """
        Initialize connection to SQLite tracking database.
        """
        self.db_path = db_path
        
        # Ensure parent directory physically exists if providing a file path
        db_dir = os.path.dirname(self.db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
            
        self.init_db()
        
        # Setup asynchronous write queue to prevent disk I/O from blocking CV pipeline
        self.write_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._db_worker, daemon=True)
        self.worker_thread.start()

    def _get_connection(self):
        """Helper to return a configured SQLite connection."""
        if self.db_path == ":memory:":
            conn = sqlite3.connect("file::memory:?cache=shared", uri=True, timeout=10)
        else:
            conn = sqlite3.connect(self.db_path, timeout=10)
            
        # Enable Write-Ahead Logging (WAL) and NORMAL synchronous mode for high concurrency and performance
        conn.execute('PRAGMA journal_mode=WAL;')
        conn.execute('PRAGMA synchronous=NORMAL;')
        return conn

    def init_db(self):
        """Create the schema securely if it does not already exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS inventory_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        object_id INTEGER,
                        event_type TEXT,
                        count_after_event INTEGER
                    )
                ''')
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS security_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                        alert_type TEXT,
                        object_id INTEGER,
                        last_zone TEXT
                    )
                ''')
                conn.commit()
                logger.info(f"Database schema initialized accurately at {self.db_path} (WAL mode enabled)")
        except Exception as e:
            logger.error(f"Failed to initialize database schema: {e}")

    def _db_worker(self):
        """Background thread logic for processing database writes sequentially."""
        while True:
            task = self.write_queue.get()
            if task is None: # Poison pill for graceful shutdown if ever needed
                break
                
            task_type, data = task
            try:
                with self._get_connection() as conn:
                    cursor = conn.cursor()
                    if task_type == 'event':
                        cursor.execute(
                            '''INSERT INTO inventory_events (timestamp, object_id, event_type, count_after_event) 
                               VALUES (?, ?, ?, ?)''',
                            data
                        )
                    elif task_type == 'alert':
                        cursor.execute(
                            '''INSERT INTO security_alerts (timestamp, alert_type, object_id, last_zone) 
                               VALUES (?, ?, ?, ?)''',
                            data
                        )
                    conn.commit()
            except Exception as e:
                logger.error(f"Async DB Write Failure for {task_type}: {e}")
            finally:
                self.write_queue.task_done()

    def insert_event(self, object_id, event_type, count_after_event):
        """
        Queue an event for asynchronous disk write.
        """
        data = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), object_id, event_type, count_after_event)
        self.write_queue.put(('event', data))

    def get_current_count(self):
        """
        Fetch the active aggregate count by checking the mathematically accurate terminal entry on the ledger.
        Defaults securely to 0.
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT count_after_event FROM inventory_events ORDER BY id DESC LIMIT 1')
                result = cursor.fetchone()
                return result[0] if result else 0
        except Exception as e:
            logger.error(f"Failed to cleanly resurrect count from DB. Defaulting to 0: {e}")
            return 0

    def insert_alert(self, alert_type, object_id, last_zone):
        """Queue suspicious anomalies for asynchronous persistence."""
        data = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), alert_type, object_id, last_zone)
        self.write_queue.put(('alert', data))

    def get_recent_alerts(self, limit=10):
        """Fetch the most recent critical security alerts for frontend UI."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT timestamp, alert_type, object_id, last_zone FROM security_alerts ORDER BY id DESC LIMIT ?', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch security alerts: {e}")
            return []

    def get_recent_events(self, limit=10):
        """Fetch the most recent inventory IN/OUT logs for frontend UI."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('SELECT timestamp, event_type, object_id, count_after_event FROM inventory_events ORDER BY id DESC LIMIT ?', (limit,))
                return cursor.fetchall()
        except Exception as e:
            logger.error(f"Failed to fetch inventory events: {e}")
            return []
