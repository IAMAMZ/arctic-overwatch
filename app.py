#!/usr/bin/env python3
"""
Minimal Flask backend for SAR Vessel Detection
Serves only the map and API endpoints, no UI elements
"""

from flask import Flask, render_template_string, jsonify, request
from flask_cors import CORS
import pandas as pd
import json
import os
from pathlib import Path
import tempfile
import uuid
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'tif', 'tiff', 'geotiff'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Ensure upload directory exists
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_detections():
    """Load detection data from CSV file."""
    csv_path = 'myoutput/detections.csv'
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        detections = []
        for idx, row in df.iterrows():
            detection = {
                'id': idx + 1,
                'lat': float(row.get('lat', 0)) if pd.notna(row.get('lat')) else 0,
                'lon': float(row.get('lon', 0)) if pd.notna(row.get('lon')) else 0,
                'intensity_db': float(row.get('intensity_db', 0)) if pd.notna(row.get('intensity_db')) else 0,
                'area_pixels': int(row.get('area_pixels', 0)) if pd.notna(row.get('area_pixels')) else 0,
                'has_ais_match': bool(row.get('has_ais_match', False)) if pd.notna(row.get('has_ais_match')) else False,
                'nearest_mmsi': str(row.get('nearest_mmsi', '')) if pd.notna(row.get('nearest_mmsi')) else None,
                'nearest_distance_m': float(row.get('nearest_distance_m', 0)) if pd.notna(row.get('nearest_distance_m')) else None,
                'nearest_time_delta_s': float(row.get('nearest_time_delta_s', 0)) if pd.notna(row.get('nearest_time_delta_s')) else None,
                'ais_timestamp': str(row.get('ais_timestamp', '')) if pd.notna(row.get('ais_timestamp')) else None,
                'row': float(row.get('row', 0)) if pd.notna(row.get('row')) else 0,
                'col': float(row.get('col', 0)) if pd.notna(row.get('col')) else 0
            }
            
            # Only include detections with valid coordinates
            if detection['lat'] != 0 and detection['lon'] != 0:
                detections.append(detection)
        
        return detections
    
    except Exception as e:
        print(f"Error loading detections: {e}")
        return []

# Minimal HTML template for map-only view
MAP_ONLY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Arctic Overwatch Map</title>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css" />
    <style>
        body, html { 
            margin: 0; 
            padding: 0; 
            height: 100%; 
            overflow: hidden; 
        }
        #map { 
            height: 100vh; 
            width: 100vw; 
        }
        .detection-marker {
            display: flex;
            justify-content: center;
            align-items: center;
        }
    </style>
</head>
<body>
    <div id="map"></div>
    
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <script>
        // Global variables
        let map;
        let detectionsLayer;
        let detections = [];

        // Initialize the application
        document.addEventListener('DOMContentLoaded', function() {
            initializeMap();
            loadDetections();
        });

        function initializeMap() {
            // Initialize the map - default to Arctic region
            map = L.map('map').setView([81, -20], 6);
            
            // Add multiple tile layers
            const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
                attribution: '© OpenStreetMap contributors'
            });
            
            const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
                attribution: 'Tiles © Esri'
            });
            
            const darkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '© OpenStreetMap contributors © CARTO'
            });
            
            // Add default layer
            satelliteLayer.addTo(map);
            
            // Layer control
            const baseLayers = {
                "Satellite": satelliteLayer,
                "OpenStreetMap": osmLayer,
                "Dark": darkLayer
            };
            
            L.control.layers(baseLayers).addTo(map);
            
            // Initialize detections layer
            detectionsLayer = L.layerGroup().addTo(map);
            
            // Add scale control
            L.control.scale().addTo(map);
        }

        function loadDetections() {
            fetch('/api/detections')
                .then(response => response.json())
                .then(data => {
                    if (data.detections) {
                        detections = data.detections;
                        updateMap();
                    }
                })
                .catch(error => {
                    console.error('Error loading detections:', error);
                });
        }

        function updateMap() {
            // Clear existing markers
            detectionsLayer.clearLayers();
            
            if (detections.length === 0) return;
            
            // Create custom icons
            const matchedIcon = L.divIcon({
                className: 'detection-marker matched',
                html: '<i class="fas fa-ship" style="color: #28a745; font-size: 16px;"></i>',
                iconSize: [25, 25],
                iconAnchor: [12, 12]
            });
            
            const unmatchedIcon = L.divIcon({
                className: 'detection-marker unmatched',
                html: '<i class="fas fa-ship" style="color: #dc3545; font-size: 16px;"></i>',
                iconSize: [25, 25],
                iconAnchor: [12, 12]
            });
            
            const darkVesselIcon = L.divIcon({
                className: 'detection-marker dark',
                html: '<i class="fas fa-user-secret" style="color: #ffc107; font-size: 16px;"></i>',
                iconSize: [25, 25],
                iconAnchor: [12, 12]
            });
            
            // Add markers for each detection
            detections.forEach(detection => {
                if (detection.lat && detection.lon) {
                    let icon;
                    if (detection.has_ais_match) {
                        icon = matchedIcon;
                    } else {
                        icon = detection.intensity_db > 25 ? darkVesselIcon : unmatchedIcon;
                    }
                    
                    const marker = L.marker([detection.lat, detection.lon], {icon})
                        .bindPopup(createPopupContent(detection));
                    
                    detectionsLayer.addLayer(marker);
                }
            });
            
            // Fit map to detections if any exist
            if (detections.length > 0 && detections.some(d => d.lat && d.lon)) {
                const group = new L.featureGroup(detectionsLayer.getLayers());
                map.fitBounds(group.getBounds().pad(0.1));
            }
        }

        function createPopupContent(detection) {
            const matchStatus = detection.has_ais_match ? 
                '<span style="color: #28a745; font-weight: bold;">AIS Match</span>' : 
                '<span style="color: #dc3545; font-weight: bold;">No AIS Match</span>';
            
            const distance = detection.nearest_distance_m ? 
                `${Math.round(detection.nearest_distance_m)}m` : 'N/A';
            
            return `
                <div style="min-width: 200px;">
                    <h6><i class="fas fa-ship"></i> Detection #${detection.id}</h6>
                    <p><strong>Position:</strong> ${detection.lat.toFixed(4)}°N, ${Math.abs(detection.lon).toFixed(4)}°W</p>
                    <p><strong>Intensity:</strong> ${detection.intensity_db.toFixed(1)} dB</p>
                    <p><strong>Area:</strong> ${detection.area_pixels} pixels</p>
                    <p><strong>Status:</strong> ${matchStatus}</p>
                    ${detection.nearest_mmsi ? `<p><strong>Nearest MMSI:</strong> ${detection.nearest_mmsi}</p>` : ''}
                    <p><strong>Distance to AIS:</strong> ${distance}</p>
                </div>
            `;
        }

        // Refresh function for external calls
        window.refreshMap = function() {
            loadDetections();
        };
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the minimal map-only page."""
    return render_template_string(MAP_ONLY_TEMPLATE)

@app.route('/api/detections')
def get_detections():
    """API endpoint to get detection data."""
    detections = load_detections()
    return jsonify({
        'success': True,
        'detections': detections,
        'count': len(detections)
    })

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle file uploads and process SAR images."""
    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Get parameters
            sigma = float(request.form.get('sigma', 3.0))
            window = int(request.form.get('window', 25))
            min_area_pixels = int(request.form.get('min_area_pixels', 5))
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Here you would typically process the SAR image
            # For now, we'll return the existing detections
            # You can integrate your SAR processing logic here
            
            # Process the file (placeholder - integrate your SAR processing here)
            success = process_sar_image(filepath, sigma, window, min_area_pixels)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if success:
                # Reload detections after processing
                detections = load_detections()
                return jsonify({
                    'success': True,
                    'message': 'File processed successfully',
                    'detections': detections,
                    'count': len(detections)
                })
            else:
                return jsonify({'success': False, 'error': 'Processing failed'}), 500
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

def process_sar_image(filepath, sigma, window, min_area_pixels):
    """
    Process SAR image and generate detections.
    Replace this with your actual SAR processing logic.
    """
    try:
        # Placeholder for your SAR processing
        # You should integrate your existing SAR detection code here
        
        # For now, just return True to indicate successful processing
        # Your actual implementation would:
        # 1. Load the SAR image from filepath
        # 2. Apply your detection algorithm with the given parameters
        # 3. Save results to myoutput/detections.csv
        # 4. Return True if successful
        
        print(f"Processing SAR image: {filepath}")
        print(f"Parameters: sigma={sigma}, window={window}, min_area={min_area_pixels}")
        
        # Import and use your SAR processing modules here
        # Example:
        # from sar_app.detection import process_image
        # success = process_image(filepath, sigma, window, min_area_pixels)
        
        return True  # Placeholder return
        
    except Exception as e:
        print(f"Error processing SAR image: {e}")
        return False

@app.route('/api/stats')
def get_stats():
    """Get detection statistics."""
    detections = load_detections()
    
    total_detections = len(detections)
    ais_matches = sum(1 for d in detections if d['has_ais_match'])
    dark_vessels = total_detections - ais_matches
    avg_intensity = sum(d['intensity_db'] for d in detections) / total_detections if total_detections > 0 else 0
    
    return jsonify({
        'total_detections': total_detections,
        'ais_matches': ais_matches,
        'dark_vessels': dark_vessels,
        'avg_intensity': round(avg_intensity, 1)
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)