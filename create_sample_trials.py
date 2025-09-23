#!/usr/bin/env python3
"""
Script to create a variety of fictional ARV trials for homepage demonstration
"""

import sqlite3
import json
from datetime import datetime, timedelta
import random

def create_sample_trials():
    """Create diverse fictional trials to showcase different domains and stages"""
    conn = sqlite3.connect('arv.db')
    cursor = conn.cursor()
    
    try:
        # Get admin user ID (assuming first admin user)
        cursor.execute("SELECT id FROM user WHERE role = 'admin' LIMIT 1")
        admin_result = cursor.fetchone()
        if not admin_result:
            print("No admin user found. Creating one...")
            cursor.execute("""
                INSERT INTO user (name, email, password_hash, role, created_at)
                VALUES (?, ?, ?, ?, datetime('now'))
            """, ('ARV_Admin', 'admin@consenses.arv', 'hashed_password', 'admin'))
            admin_id = cursor.lastrowid
        else:
            admin_id = admin_result[0]
        
        # Get some target IDs for assignment
        cursor.execute("SELECT id FROM target LIMIT 20")
        target_ids = [row[0] for row in cursor.fetchall()]
        if len(target_ids) < 6:
            print("Warning: Not enough targets available. Some trials may not have complete target assignments.")
        
        # Sample trials with different domains and statuses
        sample_trials = [
            {
                'title': 'Bitcoin Price Movement - End of August 2025',
                'domain': 'stocks',
                'event_spec_json': json.dumps({'description': 'Will Bitcoin be above or below $65,000 on August 31, 2025?'}),
                'result_time_utc': datetime.now() + timedelta(days=19),
                'status': 'open',
                'market_type': 'binary'
            },
            {
                'title': 'UEFA Champions League Final 2026',
                'domain': 'sports',
                'event_spec_json': json.dumps({'description': 'Which team will win the UEFA Champions League Final in 2026?', 'team_a': 'Manchester City', 'team_b': 'Real Madrid'}),
                'result_time_utc': datetime.now() + timedelta(days=365),
                'status': 'open',
                'market_type': 'ternary'
            },
            {
                'title': 'US Presidential Election 2028 - Early Prediction',
                'domain': 'stocks',
                'event_spec_json': json.dumps({'description': 'Which party will win the US Presidential Election in 2028?'}),
                'result_time_utc': datetime.now() + timedelta(days=1460),
                'status': 'draft',
                'market_type': 'binary'
            },
            {
                'title': 'Mega Millions Lottery - Next Drawing Color',
                'domain': 'lottery',
                'event_spec_json': json.dumps({'description': 'What will be the dominant color theme of the next Mega Millions winning numbers visualization?', 'num_balls': 5}),
                'result_time_utc': datetime.now() + timedelta(days=3),
                'status': 'live',
                'market_type': 'lottery'
            },
            {
                'title': 'Stock Market Volatility - September 2025',
                'domain': 'stocks',
                'event_spec_json': json.dumps({'description': 'Will the VIX (Volatility Index) reach above 30 during September 2025?'}),
                'result_time_utc': datetime.now() + timedelta(days=50),
                'status': 'settled',
                'market_type': 'binary'
            },
            {
                'title': 'Climate Event - Hurricane Season 2025',
                'domain': 'lottery',
                'event_spec_json': json.dumps({'description': 'How many Category 4+ hurricanes will occur in the Atlantic in 2025?', 'num_balls': 4}),
                'result_time_utc': datetime.now() + timedelta(days=120),
                'status': 'open',
                'market_type': 'lottery'
            },
            {
                'title': 'Technology Breakthrough - Quantum Computing',
                'domain': 'stocks',
                'event_spec_json': json.dumps({'description': 'Will a major quantum computing breakthrough be announced by a tech giant before 2026?'}),
                'result_time_utc': datetime.now() + timedelta(days=500),
                'status': 'open',
                'market_type': 'binary'
            },
            {
                'title': 'World Cup 2026 - Host Nation Performance',
                'domain': 'sports',
                'event_spec_json': json.dumps({'description': 'How far will the USA reach in the 2026 FIFA World Cup?', 'team_a': 'Quarter-finals+', 'team_b': 'Round of 16 or worse'}),
                'result_time_utc': datetime.now() + timedelta(days=700),
                'status': 'draft',
                'market_type': 'ternary'
            }
        ]
        
        # Insert trials
        for trial_data in sample_trials:
            cursor.execute("""
                INSERT INTO trial (
                    title, domain, event_spec_json, result_time_utc, status, 
                    market_type, decision_rule_json, created_by, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
            """, (
                trial_data['title'],
                trial_data['domain'],
                trial_data['event_spec_json'],
                trial_data['result_time_utc'].isoformat(),
                trial_data['status'],
                trial_data['market_type'],
                '{}',
                admin_id
            ))
            
            trial_id = cursor.lastrowid
            print(f"Created trial: {trial_data['title']} (ID: {trial_id}, Status: {trial_data['status']})")
            
            # Create trial outcomes based on market type
            if trial_data['market_type'] == 'binary':
                outcomes = ['A', 'B']
            elif trial_data['market_type'] == 'ternary':
                outcomes = ['A', 'B', 'C'] 
            else:  # lottery
                event_spec = json.loads(trial_data['event_spec_json'])
                num_balls = event_spec.get('num_balls', 4)
                outcomes = [f'Ball {i+1}' for i in range(num_balls)]
            
            for i, label in enumerate(outcomes):
                # Assign targets if available
                target_id = target_ids[i % len(target_ids)] if target_ids else None
                
                cursor.execute("""
                    INSERT INTO trialoutcome (trial_id, label, implied_prob)
                    VALUES (?, ?, ?)
                """, (trial_id, label, 1.0/len(outcomes)))
                
                outcome_id = cursor.lastrowid
                
                # Link target to trial outcome if target exists
                if target_id:
                    cursor.execute("""
                        INSERT INTO trialtarget (trial_id, outcome_id, target_id)
                        VALUES (?, ?, ?)
                    """, (trial_id, outcome_id, target_id))
        
        conn.commit()
        
        # Show summary
        cursor.execute("SELECT COUNT(*) FROM trial")
        total_trials = cursor.fetchone()[0]
        
        cursor.execute("SELECT status, COUNT(*) FROM trial GROUP BY status")
        status_counts = cursor.fetchall()
        
        print(f"\nSample trials created successfully!")
        print(f"Total trials in database: {total_trials}")
        print("Trial status breakdown:")
        for status, count in status_counts:
            print(f"  {status}: {count}")
        
    except Exception as e:
        print(f"Error creating sample trials: {e}")
        conn.rollback()
    finally:
        conn.close()

if __name__ == "__main__":
    create_sample_trials()