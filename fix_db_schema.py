import sqlite3
import os

# Connect to the SQLite database
db_path = "arv.db"
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

try:
    # Add created_by column to trial table if it doesn't exist
    cursor.execute("ALTER TABLE trial ADD COLUMN created_by INTEGER")
    print("Added created_by column to trial table")
except sqlite3.OperationalError as e:
    if "duplicate column name" in str(e):
        print("created_by column already exists")
    else:
        print(f"Error adding created_by column: {e}")

# Commit changes and close connection
conn.commit()
conn.close()

print("Database schema update complete")