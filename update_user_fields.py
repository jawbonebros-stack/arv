#!/usr/bin/env python3
"""
Database migration script to add user statistics fields for prediction points and leaderboard
Run this once to update existing users with the new fields
"""

import sqlite3
from sqlmodel import SQLModel, create_engine, Session
from main import User, Prediction

def update_database():
    # Connect directly to SQLite to add columns
    conn = sqlite3.connect('arv.db')
    cursor = conn.cursor()
    
    try:
        # Add new columns to User table if they don't exist
        # Check if table exists first
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cursor.fetchone():
            print("User table doesn't exist yet - it will be created automatically")
            conn.close()
            return
            
        cursor.execute("PRAGMA table_info(user)")
        columns = [row[1] for row in cursor.fetchall()]
        
        if 'total_points' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN total_points INTEGER DEFAULT 0")
            print("Added total_points column to User table")
            
        if 'total_predictions' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN total_predictions INTEGER DEFAULT 0")
            print("Added total_predictions column to User table")
            
        if 'correct_predictions' not in columns:
            cursor.execute("ALTER TABLE user ADD COLUMN correct_predictions INTEGER DEFAULT 0")
            print("Added correct_predictions column to User table")
        
        conn.commit()
        print("Database schema updated successfully!")
        
    except Exception as e:
        print(f"Error updating database: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    update_database()