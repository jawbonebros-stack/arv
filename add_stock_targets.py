#!/usr/bin/env python3
"""
Add 100 additional photo-realistic stock images as ARV targets.
Each target is carefully selected for maximum visual distinctiveness to enhance remote viewing discrimination.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import Target, Session, create_engine

# 100 Additional Photo-Realistic Stock Targets
# Each target selected for maximum distinctiveness and ARV suitability
ADDITIONAL_TARGETS = [
    # Animals & Wildlife (20 targets)
    {"tags": "elephant, large, gray, trunk, tusks, african, majestic", "uri": "/static/elephant.jpg"},
    {"tags": "dolphin, blue, ocean, jumping, marine, intelligent, sleek", "uri": "/static/dolphin.jpg"},
    {"tags": "tiger, orange, stripes, fierce, predator, jungle, powerful", "uri": "/static/tiger.jpg"},
    {"tags": "peacock, colorful, feathers, display, blue, green, ornate", "uri": "/static/peacock.jpg"},
    {"tags": "owl, nocturnal, wise, brown, feathers, large eyes, silent", "uri": "/static/owl.jpg"},
    {"tags": "penguin, black, white, antarctic, waddle, cold, formal", "uri": "/static/penguin.jpg"},
    {"tags": "giraffe, tall, spotted, neck, african, savanna, graceful", "uri": "/static/giraffe.jpg"},
    {"tags": "shark, predator, teeth, ocean, fear, dangerous, sleek", "uri": "/static/shark.jpg"},
    {"tags": "butterfly, monarch, orange, wings, delicate, transformation, flight", "uri": "/static/monarch.jpg"},
    {"tags": "horse, galloping, mane, powerful, freedom, wild, brown", "uri": "/static/horse.jpg"},
    {"tags": "snake, coiled, scales, reptile, dangerous, green, sinuous", "uri": "/static/snake.jpg"},
    {"tags": "eagle, soaring, wings, predator, sharp, freedom, majestic", "uri": "/static/eagle.jpg"},
    {"tags": "whale, massive, ocean, spout, gentle, giant, blue", "uri": "/static/whale.jpg"},
    {"tags": "flamingo, pink, long legs, tropical, elegant, curved neck", "uri": "/static/flamingo.jpg"},
    {"tags": "spider, web, eight legs, intricate, predator, geometric", "uri": "/static/spider.jpg"},
    {"tags": "lion, mane, roar, king, fierce, golden, pride", "uri": "/static/lion.jpg"},
    {"tags": "frog, green, pond, amphibian, jump, wet, lily pad", "uri": "/static/frog.jpg"},
    {"tags": "bear, brown, forest, powerful, hibernation, fur, claws", "uri": "/static/bear.jpg"},
    {"tags": "rabbit, white, fluffy, ears, hop, soft, innocent", "uri": "/static/rabbit.jpg"},
    {"tags": "turtle, shell, slow, ancient, green, water, protection", "uri": "/static/turtle.jpg"},

    # Transportation & Vehicles (15 targets)
    {"tags": "airplane, flight, wings, sky, travel, speed, metal", "uri": "/static/airplane.jpg"},
    {"tags": "motorcycle, speed, leather, road, engine, chrome, freedom", "uri": "/static/motorcycle.jpg"},
    {"tags": "sailboat, ocean, wind, white, peaceful, mast, adventure", "uri": "/static/sailboat.jpg"},
    {"tags": "train, locomotive, steam, tracks, power, journey, iron", "uri": "/static/train.jpg"},
    {"tags": "bicycle, pedals, wheels, exercise, simple, chain, green", "uri": "/static/bicycle.jpg"},
    {"tags": "helicopter, rotor, flight, rescue, noise, versatile, hovering", "uri": "/static/helicopter.jpg"},
    {"tags": "truck, cargo, heavy, diesel, transport, large, powerful", "uri": "/static/truck.jpg"},
    {"tags": "submarine, underwater, periscope, deep, exploration, steel", "uri": "/static/submarine.jpg"},
    {"tags": "hot air balloon, colorful, float, basket, peaceful, sky", "uri": "/static/balloon.jpg"},
    {"tags": "race car, speed, aerodynamic, competition, bright, fast", "uri": "/static/racecar.jpg"},
    {"tags": "school bus, yellow, children, safety, large, community", "uri": "/static/schoolbus.jpg"},
    {"tags": "fire truck, red, emergency, ladder, hero, rescue, sirens", "uri": "/static/firetruck.jpg"},
    {"tags": "tractor, farm, green, agriculture, soil, powerful, rural", "uri": "/static/tractor.jpg"},
    {"tags": "yacht, luxury, white, ocean, wealth, leisure, elegant", "uri": "/static/yacht.jpg"},
    {"tags": "skateboard, wheels, youth, tricks, street, balance, freedom", "uri": "/static/skateboard.jpg"},

    # Food & Beverages (15 targets)
    {"tags": "pizza, cheese, italian, round, delicious, pepperoni, hot", "uri": "/static/pizza_slice.jpg"},
    {"tags": "hamburger, beef, bun, american, layers, sauce, comfort", "uri": "/static/hamburger.jpg"},
    {"tags": "sushi, japanese, fish, rice, art, fresh, precise", "uri": "/static/sushi.jpg"},
    {"tags": "ice cream, cold, sweet, cone, summer, colorful, treat", "uri": "/static/icecream.jpg"},
    {"tags": "wine, red, grape, glass, elegant, aged, romantic", "uri": "/static/wine.jpg"},
    {"tags": "chocolate, sweet, brown, rich, dessert, indulgent, smooth", "uri": "/static/chocolate.jpg"},
    {"tags": "bread, loaf, golden, bakery, warm, crust, nourishment", "uri": "/static/bread.jpg"},
    {"tags": "strawberry, red, sweet, seeds, summer, fresh, juicy", "uri": "/static/strawberry.jpg"},
    {"tags": "cheese, yellow, aged, holes, dairy, sharp, traditional", "uri": "/static/cheese.jpg"},
    {"tags": "coffee, dark, beans, hot, energy, aroma, morning", "uri": "/static/coffee_beans.jpg"},
    {"tags": "donut, glazed, sweet, round, colorful, sprinkles, treat", "uri": "/static/donut.jpg"},
    {"tags": "lobster, red, claws, seafood, expensive, ocean, luxury", "uri": "/static/lobster.jpg"},
    {"tags": "avocado, green, healthy, creamy, pit, nutrition, fresh", "uri": "/static/avocado.jpg"},
    {"tags": "pancakes, stack, syrup, breakfast, fluffy, golden, comfort", "uri": "/static/pancakes.jpg"},
    {"tags": "popcorn, white, movie, kernels, snack, fluffy, entertainment", "uri": "/static/popcorn.jpg"},

    # Technology & Electronics (15 targets)
    {"tags": "smartphone, screen, modern, communication, touch, sleek, smart", "uri": "/static/smartphone.jpg"},
    {"tags": "laptop, computer, keyboard, work, portable, screen, technology", "uri": "/static/laptop.jpg"},
    {"tags": "camera, lens, photography, capture, memories, flash, focus", "uri": "/static/camera.jpg"},
    {"tags": "headphones, music, audio, sound, black, wireless, immersion", "uri": "/static/headphones.jpg"},
    {"tags": "robot, artificial, metal, intelligence, future, mechanical, automation", "uri": "/static/robot.jpg"},
    {"tags": "television, screen, entertainment, broadcast, large, viewing, family", "uri": "/static/television.jpg"},
    {"tags": "drone, flight, propellers, remote, surveillance, modern, hovering", "uri": "/static/drone.jpg"},
    {"tags": "gaming controller, buttons, play, entertainment, ergonomic, digital", "uri": "/static/gamecontroller.jpg"},
    {"tags": "smartwatch, wrist, fitness, time, notifications, health, compact", "uri": "/static/smartwatch.jpg"},
    {"tags": "virtual reality, goggles, immersion, digital, future, experience", "uri": "/static/vr_headset.jpg"},
    {"tags": "solar panel, energy, sustainable, blue, electricity, environmental, grid", "uri": "/static/solarpanel.jpg"},
    {"tags": "microphone, audio, recording, voice, performance, sound, broadcast", "uri": "/static/microphone.jpg"},
    {"tags": "satellite, space, communication, orbit, technology, global, dish", "uri": "/static/satellite.jpg"},
    {"tags": "3d printer, creation, plastic, innovation, manufacturing, layers, maker", "uri": "/static/3dprinter.jpg"},
    {"tags": "server rack, data, computing, network, power, digital, infrastructure", "uri": "/static/serverrack.jpg"},

    # Nature & Landscapes (15 targets)
    {"tags": "volcano, lava, eruption, mountain, fire, dangerous, explosive", "uri": "/static/volcano.jpg"},
    {"tags": "waterfall, cascade, rocks, mist, power, natural, refreshing", "uri": "/static/waterfall.jpg"},
    {"tags": "desert, sand, dunes, hot, vast, barren, golden", "uri": "/static/desert.jpg"},
    {"tags": "forest, trees, green, peaceful, oxygen, wildlife, dense", "uri": "/static/forest.jpg"},
    {"tags": "ocean, waves, blue, vast, horizon, salty, endless", "uri": "/static/ocean.jpg"},
    {"tags": "canyon, red, carved, deep, ancient, layered, majestic", "uri": "/static/canyon.jpg"},
    {"tags": "glacier, ice, blue, cold, melting, ancient, massive", "uri": "/static/glacier.jpg"},
    {"tags": "lightning, storm, electric, dangerous, bright, energy, dramatic", "uri": "/static/lightning.jpg"},
    {"tags": "rainbow, colors, arc, rain, hope, spectrum, beautiful", "uri": "/static/rainbow.jpg"},
    {"tags": "cave, dark, underground, stalactites, mysterious, echo, exploration", "uri": "/static/cave.jpg"},
    {"tags": "tornado, spiral, destructive, wind, funnel, dangerous, spinning", "uri": "/static/tornado.jpg"},
    {"tags": "aurora, northern lights, green, sky, magical, polar, dancing", "uri": "/static/aurora.jpg"},
    {"tags": "coral reef, colorful, underwater, fish, tropical, living, vibrant", "uri": "/static/coral.jpg"},
    {"tags": "geyser, hot, steam, water, eruption, natural, powerful", "uri": "/static/geyser.jpg"},
    {"tags": "iceberg, white, floating, cold, hidden, massive, arctic", "uri": "/static/iceberg.jpg"},

    # Sports & Recreation (10 targets)
    {"tags": "basketball, orange, court, dribble, hoop, team, bounce", "uri": "/static/basketball.jpg"},
    {"tags": "soccer ball, black, white, kick, goal, world, round", "uri": "/static/soccerball.jpg"},
    {"tags": "tennis racket, strings, court, swing, yellow, sport, precision", "uri": "/static/tennisracket.jpg"},
    {"tags": "golf ball, white, dimples, tee, precision, green, small", "uri": "/static/golfball.jpg"},
    {"tags": "surfboard, waves, ocean, balance, sport, colorful, adventure", "uri": "/static/surfboard.jpg"},
    {"tags": "skiing, snow, slopes, winter, speed, poles, mountain", "uri": "/static/skis.jpg"},
    {"tags": "bowling, pins, strike, alley, heavy, precision, sport", "uri": "/static/bowling.jpg"},
    {"tags": "boxing gloves, red, fight, protection, sport, power, combat", "uri": "/static/boxinggloves.jpg"},
    {"tags": "chess, strategy, board, pieces, intellectual, black, white", "uri": "/static/chess.jpg"},
    {"tags": "dart board, target, precision, circles, game, accuracy, focus", "uri": "/static/dartboard.jpg"},

    # Architecture & Buildings (10 targets)
    {"tags": "lighthouse, beacon, ocean, tall, navigation, red, safety", "uri": "/static/lighthouse.jpg"},
    {"tags": "castle, medieval, stone, fortress, towers, history, majestic", "uri": "/static/castle.jpg"},
    {"tags": "skyscraper, tall, glass, urban, modern, vertical, impressive", "uri": "/static/skyscraper.jpg"},
    {"tags": "bridge, span, connection, engineering, arch, transportation, crossing", "uri": "/static/bridge.jpg"},
    {"tags": "windmill, blades, energy, rural, rotation, sustainable, tall", "uri": "/static/windmill.jpg"},
    {"tags": "pyramid, ancient, egypt, triangular, mysterious, stone, monument", "uri": "/static/pyramid.jpg"},
    {"tags": "church, steeple, faith, community, bell, spiritual, traditional", "uri": "/static/church.jpg"},
    {"tags": "barn, red, farm, rural, hay, agriculture, traditional", "uri": "/static/barn.jpg"},
    {"tags": "mansion, luxury, large, wealthy, columns, impressive, estate", "uri": "/static/mansion.jpg"},
    {"tags": "observatory, telescope, dome, stars, science, astronomy, research", "uri": "/static/observatory.jpg"}
]

def add_targets_to_database():
    """Add all 100 additional targets to the database."""
    engine = create_engine('sqlite:///arv.db')
    
    with Session(engine) as session:
        added_count = 0
        skipped_count = 0
        
        for target_data in ADDITIONAL_TARGETS:
            # Check if target already exists (by tags or URI)
            existing = session.query(Target).filter(
                (Target.tags == target_data["tags"]) | 
                (Target.uri == target_data["uri"])
            ).first()
            
            if existing:
                print(f"Skipping existing target: {target_data['tags'][:50]}...")
                skipped_count += 1
                continue
            
            # Create new target
            target = Target(
                uri=target_data["uri"],
                tags=target_data["tags"],
                modality="image"
            )
            
            session.add(target)
            added_count += 1
            print(f"Added target: {target_data['tags'][:60]}...")
        
        # Commit all changes
        session.commit()
        
        print(f"\n✅ Successfully added {added_count} new targets")
        print(f"📋 Skipped {skipped_count} existing targets")
        print(f"🎯 Total targets in database: {session.query(Target).count()}")

if __name__ == "__main__":
    print("Adding 100 additional photo-realistic stock targets...")
    print("Each target selected for maximum visual distinctiveness and ARV suitability.\n")
    add_targets_to_database()