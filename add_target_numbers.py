#!/usr/bin/env python3
"""
Migration script to add target_number column to PostgreSQL database
"""

import os
import psycopg2
import random
from dotenv import load_dotenv

load_dotenv()

def add_target_numbers():
    """Add target_number column and populate existing trials"""
    
    DATABASE_URL = os.environ.get("DATABASE_URL")
    if not DATABASE_URL:
        print("No DATABASE_URL environment variable found")
        return
    
    try:
        # Connect to PostgreSQL
        conn = psycopg2.connect(DATABASE_URL)
        cursor = conn.cursor()
        
        # Add the target_number column if it doesn't exist
        try:
            cursor.execute('ALTER TABLE trial ADD COLUMN target_number VARCHAR(7)')
            print("Added target_number column to trial table")
        except psycopg2.errors.DuplicateColumn:
            print("target_number column already exists")
        except Exception as e:
            if "already exists" in str(e):
                print("target_number column already exists")
            else:
                print(f"Error adding column: {e}")
                return
        
        # Get trials without target numbers
        cursor.execute("SELECT id FROM trial WHERE target_number IS NULL OR target_number = ''")
        trials_without_numbers = cursor.fetchall()
        
        if not trials_without_numbers:
            print("All trials already have target numbers")
            conn.close()
            return
        
        print(f"Found {len(trials_without_numbers)} trials without target numbers")
        
        # Generate unique target numbers for each trial
        for (trial_id,) in trials_without_numbers:
            # Generate unique 7-digit number
            while True:
                target_num = str(random.randint(1000000, 9999999))
                
                # Check if this number already exists
                cursor.execute("SELECT COUNT(*) FROM trial WHERE target_number = %s", (target_num,))
                if cursor.fetchone()[0] == 0:
                    break
            
            # Update the trial with the new target number
            cursor.execute("UPDATE trial SET target_number = %s WHERE id = %s", (target_num, trial_id))
            print(f"Updated trial {trial_id} with target number {target_num}")
        
        # Commit changes
        conn.commit()
        print(f"Successfully updated {len(trials_without_numbers)} trials with unique target numbers")
        
        # Close connection
        conn.close()
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    add_target_numbers()