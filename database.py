
import sqlite3
from datetime import datetime

DB_NAME = "traffic_stats.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS traffic_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            vehicle_count INTEGER,
            total_vehicles INTEGER,
            avg_speed REAL,
            traffic_status TEXT
        )
    ''')
    # Migration: Add total_vehicles if it doesn't exist
    try:
        cursor.execute('ALTER TABLE traffic_logs ADD COLUMN total_vehicles INTEGER DEFAULT 0')
    except sqlite3.OperationalError:
        pass # Already exists
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS incident_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            type TEXT,
            description TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_traffic(vehicle_count, total_vehicles, avg_speed, traffic_status):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO traffic_logs (vehicle_count, total_vehicles, avg_speed, traffic_status)
        VALUES (?, ?, ?, ?)
    ''', (vehicle_count, total_vehicles, avg_speed, traffic_status))
    conn.commit()
    conn.close()

def log_incident(incident_type, description):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO incident_logs (type, description)
        VALUES (?, ?)
    ''', (incident_type, description))
    conn.commit()
    conn.close()

def get_recent_logs(limit=20):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, vehicle_count, avg_speed FROM traffic_logs 
        ORDER BY timestamp DESC LIMIT ?
    ''', (limit,))
    logs = cursor.fetchall()
    conn.close()
    return logs[::-1] # Return in chronological order

def get_recent_incidents(limit=10):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        SELECT timestamp, type, description FROM incident_logs 
        ORDER BY timestamp DESC LIMIT ?
    ''', (limit,))
    incidents = cursor.fetchall()
    conn.close()
    return incidents
