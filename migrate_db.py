#!/usr/bin/env python3
"""
Database migration script to add category column to consensus_descriptor table
"""

import sqlite3
import os

def migrate_database():
    db_path = "arv.db"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found!")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Check if category column exists
        cursor.execute("PRAGMA table_info(consensusdescriptor)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'category' not in columns:
            print("Adding category column to consensusdescriptor table...")
            cursor.execute("ALTER TABLE consensusdescriptor ADD COLUMN category TEXT DEFAULT 'general'")
            conn.commit()
            print("✅ Successfully added category column!")
        else:
            print("Category column already exists.")
            
        # Verify the migration
        cursor.execute("PRAGMA table_info(consensusdescriptor)")
        columns = cursor.fetchall()
        print("\nCurrent table structure:")
        for column in columns:
            print(f"  {column[1]} ({column[2]})")
            
    except Exception as e:
        print(f"Error during migration: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    migrate_database()