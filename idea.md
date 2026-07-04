# 💡 Idea Proposal: AI-Powered Inventory Intelligence for Cement Warehouses

This document outlines the core problem statement, target environment, and the technical solution provided by the AI-powered video analytics system.

---

## 📌 Problem Statement

In traditional cement retail stores and warehouses, inventory management is highly manual, inefficient, and prone to discrepancy. The specific challenges include:

1. **Manual Tallying Errors**: Workers carry heavy cement bags into and out of storage in rapid succession. Manually counting these bags is exhausting, error-prone, and leads to incorrect daily stock records.
2. **Inventory Leakage & Theft**: Cement warehouses often suffer from undocumented stock movements (leakage) or direct theft, particularly during off-hours or busy periods.
3. **Lack of Auditable Timelines**: Store owners do not have a chronological, transparent ledger showing exactly when bags were loaded or unloaded, making it difficult to reconcile anomalies with specific employee shifts or delivery arrivals.
4. **Physical Occlusions & Fast Pace**: Workers walk close together, carrying bags stacked on their shoulders or backs. Standard camera solutions easily double-count, lose track, or fail to detect bags during these high-occlusion conditions.

---

## 🛠️ The Solution

This project implements an edge-computing AI system that serves as the single source of truth for inventory. By analyzing a single CCTV stream mounted at the storage room entrance, it automates stock tallying and alert logging.

### 1. Automated Counting & Tracking
* **Computer Vision Core**: The pipeline utilizes custom-trained **YOLOv8** object detection to locate cement bags.
* **Temporal Association**: **ByteTrack** ensures that once a bag is detected, it is assigned a persistent tracking ID that follows it across frames, preventing multiple counts for a single bag.
* **Tripwire Logic**: A spatial line-crossing boundary checks the trajectory of tracking IDs to determine directionality:
  * **IN**: Increment the inventory ledger (workers bringing bags into storage).
  * **OUT**: Decrement the inventory ledger (workers carrying bags out to customers).

### 2. Real-Time Operations Monitoring
* **Streamlit Dashboard**: A dashboard designed for the owner's desktop in the office. It updates dynamically, presenting:
  * Current stock status.
  * Latest stock movements (IN/OUT log with ID references).
  * A chronological timeline of suspicious event logs.

### 3. Smart Threat Detection (Anomaly Alerts)
* **Vanishing Object Detection**: If a cement bag is detected inside the storage zone and suddenly disappears from tracking *without* crossing the exit tripwire, the system flags a "Missing Object" anomaly.
* **SMTP Notification**: The anomaly engine queues a secure threat event in the local SQLite database and dispatches email alerts to the owner.

### 4. Edge-Grade Resilience (Phase 1 Target)
* **Zero Frame-Drop Database**: Uses an asynchronous thread pool to write to SQLite, ensuring disk latency never blocks the frame analyzer.
* **Write-Ahead Logging (WAL)**: Ensures the database is crash-resilient and never locks or corrupts during sudden power outages or concurrent Streamlit dashboard refreshes.
* **Auto-Reconnect**: Survives physical network/CCTV drops by retrying connections gracefully in the background.
