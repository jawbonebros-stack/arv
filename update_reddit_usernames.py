#!/usr/bin/env python3
"""
Script to update existing users with Reddit-style usernames
"""

import sqlite3

def update_to_reddit_usernames():
    """Update existing users to use Reddit-style usernames"""
    conn = sqlite3.connect('arv.db')
    cursor = conn.cursor()
    
    try:
        # Mapping of current names to Reddit-style usernames
        username_updates = [
            ('Sarah Chen', 'PsychicSarah42'),
            ('Michael Rodriguez', 'RemoteViewer99'),
            ('Emma Thompson', 'VisionQuest2024'),
            ('James Wilson', 'ARV_Master'),
            ('Lisa Park', 'IntuitiveLisa'),
            ('David Kumar', 'FutureSightDK'),
            ('Maria Garcia', 'CosmicMaria'),
            ('Alex Johnson', 'QuantumAlex'),
            ('Rachel Green', 'ThirdEyeRachel'),
            ('Tom Brown', 'PredictorTom')
        ]
        
        # Update each user's name
        for old_name, new_username in username_updates:
            cursor.execute("UPDATE user SET name = ? WHERE name = ?", (new_username, old_name))
            if cursor.rowcount > 0:
                print(f"Updated '{old_name}' to '{new_username}'")
            else:
                print(f"User '{old_name}' not found, skipping...")
        
        conn.commit()
        
        # Show updated leaderboard
        cursor.execute("""
            SELECT name, total_points, total_predictions, correct_predictions
            FROM user 
            WHERE total_predictions > 0 
            ORDER BY total_points DESC
        """)
        users = cursor.fetchall()
        
        print(f"\nUpdated Leaderboard (Reddit-style usernames):")
        print("=" * 60)
        for i, (name, points, total, correct) in enumerate(users, 1):
            accuracy = (correct / total * 100) if total > 0 else 0
            print(f"{i:2d}. {name:<20} | {points:3d} pts | {accuracy:5.1f}% ({correct}/{total})")
        
    except Exception as e:
        print(f"Error updating usernames: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    update_to_reddit_usernames()