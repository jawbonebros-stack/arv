#!/usr/bin/env python3
"""
Populate the leaderboard with Reddit-style usernames and realistic prediction data.
This script creates sample users with varied prediction histories for demonstration.
"""

import random
import hashlib
from datetime import datetime, timezone, timedelta
from sqlmodel import Session, create_engine, select
from passlib.hash import bcrypt
import os
from dotenv import load_dotenv

load_dotenv()

# Import the models from main.py
from main import User, DB_URL, engine

# Reddit-style username components
ADJECTIVES = [
    "Mystical", "Silent", "Cosmic", "Ancient", "Hidden", "Quantum", "Ethereal", "Phantom",
    "Stellar", "Neural", "Digital", "Psychic", "Remote", "Lucid", "Astral", "Temporal",
    "Sage", "Wise", "Sharp", "Quick", "Deep", "Bright", "Clear", "Swift", "Keen",
    "Shadow", "Lunar", "Solar", "Void", "Prism", "Crystal", "Echo", "Spiral"
]

NOUNS = [
    "Viewer", "Seer", "Oracle", "Prophet", "Scanner", "Reader", "Detector", "Sensor",
    "Mind", "Eye", "Vision", "Sight", "Perception", "Awareness", "Consciousness", "Spirit",
    "Wolf", "Fox", "Raven", "Eagle", "Hawk", "Owl", "Cat", "Bear", "Tiger", "Dragon",
    "Warrior", "Scout", "Hunter", "Guide", "Seeker", "Finder", "Explorer", "Wanderer",
    "Bot", "AI", "Tech", "Crypto", "Pixel", "Data", "Code", "Byte", "Node", "Link"
]

NUMBERS = list(range(1, 9999))

def generate_reddit_username():
    """Generate a Reddit-style username with adjective + noun + optional number"""
    adjective = random.choice(ADJECTIVES)
    noun = random.choice(NOUNS)
    
    # 70% chance to add numbers, 30% without
    if random.random() < 0.7:
        number = random.choice(NUMBERS)
        return f"{adjective}{noun}{number}"
    else:
        return f"{adjective}{noun}"

def create_sample_users(count=20):
    """Create sample users with Reddit-style usernames and varied prediction stats"""
    
    users_data = []
    
    for i in range(count):
        username = generate_reddit_username()
        
        # Ensure username uniqueness
        while any(user['name'] == username for user in users_data):
            username = generate_reddit_username()
        
        # Generate realistic prediction statistics
        # Some users are very active, others moderate, some beginners
        activity_level = random.choices(
            ['high', 'medium', 'low', 'beginner'], 
            weights=[0.2, 0.4, 0.3, 0.1]
        )[0]
        
        if activity_level == 'high':
            total_predictions = random.randint(50, 200)
            accuracy_base = random.uniform(0.45, 0.75)  # 45-75% accuracy
        elif activity_level == 'medium':
            total_predictions = random.randint(15, 50)
            accuracy_base = random.uniform(0.35, 0.65)  # 35-65% accuracy
        elif activity_level == 'low':
            total_predictions = random.randint(5, 15)
            accuracy_base = random.uniform(0.30, 0.60)  # 30-60% accuracy
        else:  # beginner
            total_predictions = random.randint(1, 5)
            accuracy_base = random.uniform(0.20, 0.50)  # 20-50% accuracy
        
        # Add some randomness to accuracy
        accuracy = max(0.0, min(1.0, accuracy_base + random.uniform(-0.1, 0.1)))
        correct_predictions = int(total_predictions * accuracy)
        
        # Points calculation: base points per correct prediction + bonus for consistency
        base_points = correct_predictions * 10
        consistency_bonus = int(accuracy * 50) if total_predictions > 10 else 0
        total_points = base_points + consistency_bonus
        
        # Skill score based on accuracy and volume
        skill_score = (accuracy * 0.7) + (min(total_predictions / 100, 0.3))
        
        users_data.append({
            'name': username,
            'email': f"{username.lower()}@example.com",
            'password_hash': bcrypt.hash("password123"),  # Default password for demo
            'role': 'viewer',
            'skill_score': round(skill_score, 3),
            'total_points': total_points,
            'total_predictions': total_predictions,
            'correct_predictions': correct_predictions,
            'created_at': datetime.now(timezone.utc) - timedelta(days=random.randint(1, 90))
        })
    
    return users_data

def populate_leaderboard():
    """Add sample users to the database"""
    
    print("🚀 Populating leaderboard with Reddit-style usernames...")
    
    with Session(engine) as session:
        # Check if we already have many users (avoid duplicates)
        existing_users = session.exec(select(User)).all()
        if len(existing_users) > 10:
            print(f"⚠️  Found {len(existing_users)} existing users. Clearing non-admin users first...")
            # Keep admin users, remove others
            for user in existing_users:
                if user.role != 'admin':
                    session.delete(user)
            session.commit()
        
        # Generate sample users
        users_data = create_sample_users(25)
        
        print(f"📊 Creating {len(users_data)} users with prediction data...")
        
        for user_data in users_data:
            user = User(**user_data)
            session.add(user)
        
        session.commit()
        
        # Display created users sorted by points
        print("\n🏆 Leaderboard Preview (Top 10):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Username':<20} {'Points':<8} {'Predictions':<12} {'Accuracy':<10}")
        print("-" * 80)
        
        leaderboard = session.exec(
            select(User)
            .where(User.total_predictions > 0)
            .order_by(User.total_points.desc(), User.correct_predictions.desc())
            .limit(10)
        ).all()
        
        for i, user in enumerate(leaderboard, 1):
            accuracy = f"{user.accuracy_percentage:.1f}%"
            predictions = f"{user.correct_predictions}/{user.total_predictions}"
            print(f"{i:<4} {user.name:<20} {user.total_points:<8} {predictions:<12} {accuracy:<10}")
        
        print(f"\n✅ Successfully populated leaderboard with {len(users_data)} Reddit-style users!")
        print(f"🎯 Database now contains {len(session.exec(select(User)).all())} total users")

if __name__ == "__main__":
    populate_leaderboard()