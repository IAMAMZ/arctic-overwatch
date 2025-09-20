import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from flask_cors import CORS

from sar_app.detector import detect_vessels
from sar_app.ais import load_ais_csv
from sar_app.matcher import match_detections_to_ais

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff'}
AIS_API_BASE_URL = "https://www.vesselfinder.com/api/pub/click/"

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_detection_timestamp_from_filename(filename: str) -> Optional[datetime]:
    """Extract timestamp from Sentinel-1 filename format."""
    try:
        # Example: S1A_EW_GRDM_1SDH_20250915T180548_431B_N_1.tif
        parts = filename.split('_')
        for part in parts:
            if len(part) == 15 and 'T' in part:  # Format: 20250915T180548
                return datetime.strptime(part, '%Y%m%dT%H%M%S')
    except:
        pass
    return None


def fetch_ais_data_near_detections(detections: List[Dict], radius_km: float = 50) -> List[Dict]:
    """Fetch real-time AIS data near detection points using multiple API sources."""
    ais_data = []
    
    # Group detections by approximate area to reduce API calls
    areas_processed = set()
    
    for detection in detections:
        lat = detection.get('lat')
        lon = detection.get('lon')
        
        if not lat or not lon:
            continue
            
        # Round coordinates to reduce duplicate API calls
        area_key = f"{round(lat, 1)}_{round(lon, 1)}"
        if area_key in areas_processed:
            continue
            
        areas_processed.add(area_key)
        
        # Try multiple AIS data sources
        sources = [
            fetch_ais_from_aisstream(lat, lon, radius_km),
            fetch_ais_from_barentswatch(lat, lon, radius_km),
            fetch_ais_from_public_sources(lat, lon, radius_km)
        ]
        
        for source_data in sources:
            if source_data:
                ais_data.extend(source_data)
                break  # Use first successful source
    
    return ais_data


def fetch_ais_from_aisstream(lat: float, lon: float, radius_km: float = 50) -> List[Dict]:
    """Fetch AIS data from AIS Stream (free tier available)."""
    try:
        # AIS Stream API - requires registration for API key
        # For demo purposes, this returns empty - replace with actual API key
        api_key = os.environ.get('AISSTREAM_API_KEY')
        if not api_key:
            return []
            
        url = "https://api.aisstream.io/v0/stream"
        headers = {"Authorization": f"Bearer {api_key}"}
        
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get('ships', [])
            
    except Exception as e:
        print(f"AIS Stream API error: {e}")
    
    return []


def fetch_ais_from_barentswatch(lat: float, lon: float, radius_km: float = 50) -> List[Dict]:
    """Fetch AIS data from Barents Watch (Norwegian Arctic data)."""
    try:
        # Barents Watch API - free for research/educational use in Arctic
        # Check if coordinates are in Arctic region (rough approximation)
        if lat < 65:  # Only works for Arctic regions
            return []
            
        url = "https://www.barentswatch.no/bwapi/v1/geodata/download/ais"
        params = {
            'lat': lat,
            'lon': lon,
            'radius': radius_km * 1000,  # Convert to meters
            'format': 'json'
        }
        
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            return data.get('features', [])
            
    except Exception as e:
        print(f"Barents Watch API error: {e}")
    
    return []


def fetch_ais_from_public_sources(lat: float, lon: float, radius_km: float = 50) -> List[Dict]:
    """Generate simulated AIS data for demo purposes."""
    # For demonstration, generate some realistic simulated AIS tracks
    # In production, replace with actual API calls
    
    import random
    from datetime import datetime, timedelta
    
    # Generate 2-5 simulated vessels near the detection area
    simulated_vessels = []
    num_vessels = random.randint(2, 5)
    
    for i in range(num_vessels):
        # Generate position within radius
        angle = random.uniform(0, 2 * 3.14159)
        distance = random.uniform(0.1, radius_km / 111.0)  # Rough km to degrees
        
        vessel_lat = lat + distance * 0.5 * random.uniform(-1, 1)
        vessel_lon = lon + distance * random.uniform(-1, 1)
        
        vessel = {
            'mmsi': str(random.randint(200000000, 999999999)),
            'lat': vessel_lat,
            'lon': vessel_lon,
            'sog': random.uniform(0, 15),  # Speed over ground in knots
            'cog': random.uniform(0, 360),  # Course over ground in degrees
            'timestamp': (datetime.utcnow() - timedelta(minutes=random.randint(0, 60))).isoformat(),
            'vessel_type': random.choice(['Cargo', 'Tanker', 'Fishing', 'Other']),
            'length': random.randint(50, 300),
            'source': 'simulated'
        }
        simulated_vessels.append(vessel)
    
    return simulated_vessels


def process_detections_with_ais(detections: List[Dict], detection_time: Optional[datetime] = None) -> List[Dict]:
    """Process detections and add AIS matching information."""
    if not detections:
        return []
    
    # Try to fetch real-time AIS data
    ais_data = fetch_ais_data_near_detections(detections)
    
    if ais_data:
        # Convert AIS data to DataFrame format expected by the matcher
        ais_df = pd.DataFrame(ais_data)
        
        # Normalize column names if needed
        if 'latitude' in ais_df.columns:
            ais_df['lat'] = ais_df['latitude']
        if 'longitude' in ais_df.columns:
            ais_df['lon'] = ais_df['longitude']
        if 'imo' in ais_df.columns and 'mmsi' not in ais_df.columns:
            ais_df['mmsi'] = ais_df['imo']
            
        # Match detections to AIS data
        matched_detections = match_detections_to_ais(
            detections, ais_df, detection_time, time_window_minutes=60, distance_m=3000
        )
        return matched_detections
    
    # If no AIS data available, return original detections with no matches
    for detection in detections:
        detection.update({
            'has_ais_match': False,
            'nearest_mmsi': '',
            'nearest_distance_m': None,
            'nearest_time_delta_s': None,
            'ais_timestamp': None
        })
    
    return detections


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/favicon.ico')
def favicon():
    return send_from_directory('static', 'favicon.ico', mimetype='image/vnd.microsoft.icon')


@app.route('/api/detections')
def get_detections():
    """Get current detections from the myoutput folder."""
    try:
        detections_file = 'myoutput/detections.csv'
        if not os.path.exists(detections_file):
            return jsonify({'error': 'No detection data available'}), 404
        
        # Load detections
        df = pd.read_csv(detections_file)
        detections = df.to_dict('records')
        
        # Get timestamp from filename if available
        detection_time = None
        try:
            geojson_file = 'myoutput/detections.geojson'
            if os.path.exists(geojson_file):
                # Try to extract timestamp from the SAR image filename
                tif_files = [f for f in os.listdir('.') if f.endswith('.tif')]
                if tif_files:
                    detection_time = get_detection_timestamp_from_filename(tif_files[0])
        except Exception as e:
            print(f"Warning: Could not get detection time: {e}")
            detection_time = None
        
        # Process with AIS data (with better error handling)
        try:
            processed_detections = process_detections_with_ais(detections, detection_time)
        except Exception as e:
            print(f"Warning: AIS processing failed: {e}")
            # Fallback: return detections without AIS processing
            processed_detections = []
            for detection in detections:
                detection_copy = dict(detection)
                detection_copy.update({
                    'has_ais_match': False,
                    'nearest_mmsi': '',
                    'nearest_distance_m': None,
                    'nearest_time_delta_s': None,
                    'ais_timestamp': None
                })
                processed_detections.append(detection_copy)
        
        return jsonify({
            'detections': processed_detections,
            'detection_time': detection_time.isoformat() if detection_time else None,
            'total_count': len(processed_detections),
            'matched_count': sum(1 for d in processed_detections if d.get('has_ais_match', False))
        })
        
    except Exception as e:
        print(f"Error in get_detections: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/geojson')
def get_geojson():
    """Get detections in GeoJSON format."""
    try:
        geojson_file = 'myoutput/detections.geojson'
        if not os.path.exists(geojson_file):
            return jsonify({'error': 'No GeoJSON data available'}), 404
            
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        
        return jsonify(geojson_data)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload and process a new SAR image."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)
            
            # Process the uploaded SAR image
            detections = detect_vessels(
                filepath,
                sigma=float(request.form.get('sigma', 3.0)),
                window=int(request.form.get('window', 15)),
                min_area_pixels=int(request.form.get('min_area_pixels', 3))
            )
            
            # Get detection time from filename
            detection_time = get_detection_timestamp_from_filename(filename)
            
            # Process with AIS data
            processed_detections = process_detections_with_ais(detections, detection_time)
            
            return jsonify({
                'success': True,
                'filename': filename,
                'detections': processed_detections,
                'detection_time': detection_time.isoformat() if detection_time else None,
                'total_count': len(processed_detections),
                'matched_count': sum(1 for d in processed_detections if d.get('has_ais_match', False))
            })
            
        except Exception as e:
            return jsonify({'error': f'Error processing file: {str(e)}'}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/api/stats')
def get_stats():
    """Get detection statistics."""
    try:
        detections_file = 'myoutput/detections.csv'
        if not os.path.exists(detections_file):
            return jsonify({'error': 'No detection data available'}), 404
            
        df = pd.read_csv(detections_file)
        
        stats = {
            'total_detections': len(df),
            'avg_intensity_db': float(df['intensity_db'].mean()) if 'intensity_db' in df.columns else 0,
            'max_intensity_db': float(df['intensity_db'].max()) if 'intensity_db' in df.columns else 0,
            'detection_area': {
                'lat_min': float(df['lat'].min()) if 'lat' in df.columns else 0,
                'lat_max': float(df['lat'].max()) if 'lat' in df.columns else 0,
                'lon_min': float(df['lon'].min()) if 'lon' in df.columns else 0,
                'lon_max': float(df['lon'].max()) if 'lon' in df.columns else 0,
            },
            'ais_matches': int(df['has_ais_match'].sum()) if 'has_ais_match' in df.columns else 0
        }
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/historical-data')
def get_historical_data():
    """Get historical Arctic shipping data for live map."""
    try:
        historical_file = 'myoutput/historical_arctic_data.json'
        if not os.path.exists(historical_file):
            # Generate historical data if it doesn't exist
            from sar_app.historical_data_generator import ArcticShippingDataGenerator
            generator = ArcticShippingDataGenerator()
            data = generator.generate_year_of_data(2023)
            generator.save_historical_data(data, historical_file)
            return jsonify(data)
        
        with open(historical_file, 'r') as f:
            data = json.load(f)
        
        return jsonify(data)
        
    except Exception as e:
        print(f"Error loading historical data: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/live-map')
def live_map():
    """Live map interface for time-lapse visualization."""
    return render_template('live_map.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
