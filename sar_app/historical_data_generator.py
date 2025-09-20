"""
Historical Arctic Shipping Data Generator
Generates realistic vessel detection data for time-lapse visualization
"""

import json
import random
import math
from datetime import datetime, timedelta
from typing import List, Dict, Any
import numpy as np

class ArcticShippingDataGenerator:
    """Generates realistic Arctic shipping data for demo purposes"""
    
    def __init__(self):
        # Arctic shipping routes and patterns
        self.arctic_routes = [
            # Northwest Passage routes
            {"name": "Northwest Passage", "points": [
                (70.0, -140.0), (72.0, -130.0), (74.0, -120.0), (76.0, -110.0), 
                (78.0, -100.0), (80.0, -90.0), (82.0, -80.0), (84.0, -70.0)
            ]},
            # Northern Sea Route
            {"name": "Northern Sea Route", "points": [
                (70.0, 20.0), (72.0, 30.0), (74.0, 40.0), (76.0, 50.0), 
                (78.0, 60.0), (80.0, 70.0), (82.0, 80.0), (84.0, 90.0)
            ]},
            # Transpolar route
            {"name": "Transpolar Route", "points": [
                (75.0, -60.0), (78.0, -30.0), (81.0, 0.0), (84.0, 30.0), (87.0, 60.0)
            ]}
        ]
        
        # Vessel types and their characteristics
        self.vessel_types = [
            {"type": "Cargo", "ais_probability": 0.95, "speed_range": (8, 15), "size_range": (150, 300)},
            {"type": "Tanker", "ais_probability": 0.98, "speed_range": (10, 18), "size_range": (200, 400)},
            {"type": "Fishing", "ais_probability": 0.85, "speed_range": (5, 12), "size_range": (50, 150)},
            {"type": "Research", "ais_probability": 0.90, "speed_range": (6, 14), "size_range": (80, 200)},
            {"type": "Icebreaker", "ais_probability": 0.99, "speed_range": (8, 16), "size_range": (100, 250)},
            {"type": "Unknown", "ais_probability": 0.20, "speed_range": (6, 20), "size_range": (50, 300)}  # Dark vessels
        ]
        
        # Seasonal patterns (more activity in summer)
        self.seasonal_multipliers = {
            1: 0.3, 2: 0.2, 3: 0.3, 4: 0.4, 5: 0.6, 6: 0.8,
            7: 1.0, 8: 1.0, 9: 0.9, 10: 0.7, 11: 0.4, 12: 0.3
        }

    def generate_vessel_track(self, start_date: datetime, vessel_type: Dict, route: Dict) -> List[Dict]:
        """Generate a vessel track along a route"""
        track = []
        points = route["points"]
        
        # Calculate total journey time (days)
        total_distance = self._calculate_route_distance(points)
        avg_speed = random.uniform(*vessel_type["speed_range"])  # knots
        journey_days = total_distance / (avg_speed * 24)  # Convert to days
        
        # Generate positions along the route
        num_positions = max(10, int(journey_days * 4))  # 4 positions per day
        
        for i in range(num_positions):
            progress = i / (num_positions - 1)
            lat, lon = self._interpolate_route_position(points, progress)
            
            # Add some randomness to position
            lat += random.uniform(-0.1, 0.1)
            lon += random.uniform(-0.1, 0.1)
            
            # Calculate timestamp
            timestamp = start_date + timedelta(days=journey_days * progress)
            
            # Determine if vessel has AIS
            has_ais = random.random() < vessel_type["ais_probability"]
            
            # Generate detection data
            detection = {
                "id": f"vessel_{random.randint(1000, 9999)}",
                "timestamp": timestamp.isoformat(),
                "lat": lat,
                "lon": lon,
                "vessel_type": vessel_type["type"],
                "mmsi": f"{random.randint(200000000, 999999999)}" if has_ais else "",
                "sog": random.uniform(*vessel_type["speed_range"]),
                "cog": random.uniform(0, 360),
                "length": random.uniform(*vessel_type["size_range"]),
                "has_ais_match": has_ais,
                "intensity_db": random.uniform(20, 35),
                "area_pixels": random.randint(5, 25),
                "detection_confidence": random.uniform(0.7, 0.95),
                "weather_conditions": random.choice(["Clear", "Fog", "Snow", "Storm"]),
                "ice_conditions": random.choice(["Open Water", "Light Ice", "Heavy Ice", "Pack Ice"])
            }
            
            track.append(detection)
        
        return track

    def generate_year_of_data(self, start_year: int = 2023) -> List[Dict]:
        """Generate a full year of Arctic shipping data"""
        all_detections = []
        start_date = datetime(start_year, 1, 1)
        
        # Generate detections for each month
        for month in range(1, 13):
            month_date = datetime(start_year, month, 1)
            seasonal_multiplier = self.seasonal_multipliers[month]
            
            # Number of vessels this month (more in summer)
            num_vessels = int(random.uniform(20, 80) * seasonal_multiplier)
            
            for _ in range(num_vessels):
                # Select random vessel type and route
                vessel_type = random.choice(self.vessel_types)
                route = random.choice(self.arctic_routes)
                
                # Random start date within the month
                month_start = datetime(start_year, month, 1)
                if month == 12:
                    month_end = datetime(start_year + 1, 1, 1)
                else:
                    month_end = datetime(start_year, month + 1, 1)
                
                vessel_start = month_start + timedelta(
                    days=random.randint(0, (month_end - month_start).days - 1)
                )
                
                # Generate vessel track
                track = self.generate_vessel_track(vessel_start, vessel_type, route)
                all_detections.extend(track)
        
        # Sort by timestamp
        all_detections.sort(key=lambda x: x["timestamp"])
        
        return all_detections

    def _calculate_route_distance(self, points: List[tuple]) -> float:
        """Calculate total distance of a route in nautical miles"""
        total_distance = 0
        for i in range(len(points) - 1):
            lat1, lon1 = points[i]
            lat2, lon2 = points[i + 1]
            total_distance += self._haversine_distance(lat1, lon1, lat2, lon2)
        return total_distance

    def _interpolate_route_position(self, points: List[tuple], progress: float) -> tuple:
        """Interpolate position along a route based on progress (0-1)"""
        if progress <= 0:
            return points[0]
        if progress >= 1:
            return points[-1]
        
        # Find which segment we're in
        segment_length = 1.0 / (len(points) - 1)
        segment_index = int(progress / segment_length)
        segment_progress = (progress % segment_length) / segment_length
        
        if segment_index >= len(points) - 1:
            return points[-1]
        
        lat1, lon1 = points[segment_index]
        lat2, lon2 = points[segment_index + 1]
        
        lat = lat1 + (lat2 - lat1) * segment_progress
        lon = lon1 + (lon2 - lon1) * segment_progress
        
        return lat, lon

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance between two points in nautical miles"""
        R = 3440.065  # Earth's radius in nautical miles
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    def save_historical_data(self, data: List[Dict], filename: str = "historical_arctic_data.json"):
        """Save generated data to file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {len(data)} detections to {filename}")

    def load_historical_data(self, filename: str = "historical_arctic_data.json") -> List[Dict]:
        """Load historical data from file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"Loaded {len(data)} detections from {filename}")
            return data
        except FileNotFoundError:
            print(f"File {filename} not found. Generating new data...")
            return self.generate_year_of_data()


if __name__ == "__main__":
    generator = ArcticShippingDataGenerator()
    
    # Generate a year of data
    print("Generating historical Arctic shipping data...")
    data = generator.generate_year_of_data(2023)
    
    # Save to file
    generator.save_historical_data(data, "myoutput/historical_arctic_data.json")
    
    print(f"Generated {len(data)} vessel detections")
    print(f"Date range: {data[0]['timestamp']} to {data[-1]['timestamp']}")
    
    # Show some statistics
    vessel_types = {}
    ais_matches = 0
    for detection in data:
        vessel_type = detection['vessel_type']
        vessel_types[vessel_type] = vessel_types.get(vessel_type, 0) + 1
        if detection['has_ais_match']:
            ais_matches += 1
    
    print("\nVessel type distribution:")
    for vtype, count in vessel_types.items():
        print(f"  {vtype}: {count}")
    
    print(f"\nAIS matches: {ais_matches}/{len(data)} ({ais_matches/len(data)*100:.1f}%)")
