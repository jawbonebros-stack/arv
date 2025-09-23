#!/usr/bin/env python3
"""
Database migration script to add UserFeedback table for workflow feedback collection.
Run this script to add the new table to your existing database.
"""

import sqlite3
import os
from datetime import datetime, timezone

def migrate_database():
    """Add UserFeedback table to the existing database"""
    
    db_path = "arv.db"
    
    if not os.path.exists(db_path):
        print(f"Database {db_path} not found. Please run the main application first to create the database.")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Check if UserFeedback table already exists
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name='userfeedback'
        """)
        
        if cursor.fetchone():
            print("UserFeedback table already exists. No migration needed.")
            conn.close()
            return True
        
        # Create UserFeedback table
        cursor.execute("""
            CREATE TABLE userfeedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                workflow VARCHAR NOT NULL,
                context VARCHAR,
                overall_rating INTEGER CHECK (overall_rating >= 1 AND overall_rating <= 5),
                ease_rating INTEGER CHECK (ease_rating >= 1 AND ease_rating <= 5),
                feedback_text TEXT,
                suggestions TEXT,
                issues TEXT,
                contact_ok BOOLEAN NOT NULL DEFAULT 0,
                page_url TEXT,
                user_agent TEXT,
                created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                session_id VARCHAR,
                FOREIGN KEY (user_id) REFERENCES user (id)
            )
        """)
        
        # Create index on workflow for better query performance
        cursor.execute("""
            CREATE INDEX idx_userfeedback_workflow ON userfeedback(workflow)
        """)
        
        # Create index on created_at for time-based queries
        cursor.execute("""
            CREATE INDEX idx_userfeedback_created_at ON userfeedback(created_at)
        """)
        
        conn.commit()
        print("✅ Successfully created UserFeedback table and indexes")
        
        # Verify the table was created
        cursor.execute("SELECT sql FROM sqlite_master WHERE name='userfeedback'")
        result = cursor.fetchone()
        if result:
            print(f"✅ Table schema: {result[0]}")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error creating UserFeedback table: {str(e)}")
        return False

if __name__ == "__main__":
    print("🔄 Migrating database to add UserFeedback table...")
    success = migrate_database()
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("The feedback collection system is now ready to use.")
        print("\nFeatures added:")
        print("- User feedback collection modal on all pages")
        print("- Workflow-specific feedback tracking")
        print("- Admin dashboard at /admin/feedback")
        print("- Anonymous and authenticated feedback support")
    else:
        print("\n❌ Migration failed. Please check the error messages above.")