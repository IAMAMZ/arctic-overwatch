#!/usr/bin/env python3
"""
Enhanced Flask backend for SAR Vessel Detection
Includes detailed ship information and confidence-based visualization
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

# Regional bounds for Arctic regions (lat_min, lon_min, lat_max, lon_max)
ARCTIC_REGIONS = {
    'fram strait': [78.5, -5.0, 81.0, 10.0],
    'barents sea': [70.0, 15.0, 81.0, 55.0],
    'kara sea': [68.0, 55.0, 82.0, 100.0],
    'laptev sea': [70.0, 100.0, 82.0, 140.0],
    'east siberian sea': [68.0, 140.0, 77.0, 180.0],
    'chukchi sea': [66.0, -180.0, 77.0, -150.0],
    'beaufort sea': [69.0, -150.0, 80.0, -110.0],
    'canadian arctic archipelago': [68.0, -110.0, 83.0, -60.0],
    'greenland sea': [70.0, -20.0, 83.0, 10.0],
    'norwegian sea': [62.0, -5.0, 75.0, 15.0],
    'baffin bay': [65.0, -85.0, 78.0, -50.0],
    'davis strait': [60.0, -70.0, 68.0, -50.0],
    'lincoln sea': [82.0, -70.0, 85.0, -30.0],
    'wandel sea': [81.0, -20.0, 84.0, 20.0],
    'white sea': [64.0, 32.0, 67.0, 42.0],
    'pechora sea': [68.0, 45.0, 71.0, 65.0],
    'hudson bay': [51.0, -95.0, 70.0, -75.0],
    'hudson strait': [60.0, -85.0, 65.0, -60.0],
    'northwest passage': [68.0, -110.0, 75.0, -85.0],
    'northeast passage': [70.0, 15.0, 82.0, 180.0],
    'svalbard': [76.5, 10.0, 81.0, 35.0],
    'franz josef land': [79.5, 44.0, 82.0, 65.0],
    'novaya zemlya': [70.0, 52.0, 77.0, 69.0],
    'severnaya zemlya': [78.0, 95.0, 81.5, 108.0],
    'new siberian islands': [73.5, 135.0, 76.5, 155.0],
    'wrangel island': [70.5, -180.0, 71.5, -177.0],
    'victoria island': [69.0, -115.0, 73.5, -100.0],
    'banks island': [70.5, -125.0, 74.5, -115.0],
    'ellesmere island': [76.0, -90.0, 83.0, -70.0],
    'devon island': [74.5, -95.0, 76.5, -80.0],
    # Add some broader regions
    'arctic': [66.0, -180.0, 90.0, 180.0],
    'north pole': [85.0, -180.0, 90.0, 180.0],
    'arctic ocean': [70.0, -180.0, 90.0, 180.0],
    'north atlantic': [50.0, -70.0, 85.0, 30.0],
    'north pacific': [50.0, 140.0, 85.0, -120.0],
}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_detections():
    """Load detection data from CSV file with enhanced ship information."""
    csv_path = 'myoutput/detections.csv'
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path)
        
        # Convert to list of dictionaries
        detections = []
        for idx, row in df.iterrows():
            detection = {
                'id': int(row.get('id', idx)) if pd.notna(row.get('id')) else idx,
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
                'col': float(row.get('col', 0)) if pd.notna(row.get('col')) else 0,
                'region': str(row.get('region', '')) if pd.notna(row.get('region')) else '',
                'area_description': str(row.get('area_description', '')) if pd.notna(row.get('area_description')) else '',
                'country': str(row.get('country', '')) if pd.notna(row.get('country')) else '',
                'x': float(row.get('x', 0)) if pd.notna(row.get('x')) else 0,
                'y': float(row.get('y', 0)) if pd.notna(row.get('y')) else 0,
                # Enhanced ship information
                'confidence': float(row.get('confidence', 50.0)) if pd.notna(row.get('confidence')) else 50.0,
                'ship_type': str(row.get('ship_type', 'Unknown Vessel')) if pd.notna(row.get('ship_type')) else 'Unknown Vessel',
                'ship_model': str(row.get('ship_model', 'Unidentified')) if pd.notna(row.get('ship_model')) else 'Unidentified',
                'engine_type': str(row.get('engine_type', 'Unknown')) if pd.notna(row.get('engine_type')) else 'Unknown',
                'length_m': str(row.get('length_m', 'Unknown')) if pd.notna(row.get('length_m')) else 'Unknown',
                'beam_m': str(row.get('beam_m', 'Unknown')) if pd.notna(row.get('beam_m')) else 'Unknown',
                'draft_m': str(row.get('draft_m', 'Unknown')) if pd.notna(row.get('draft_m')) else 'Unknown',
                'tonnage_gt': str(row.get('tonnage_gt', 'Unknown')) if pd.notna(row.get('tonnage_gt')) else 'Unknown',
                'flag_state': str(row.get('flag_state', 'Unknown')) if pd.notna(row.get('flag_state')) else 'Unknown',
                'vessel_name': str(row.get('vessel_name', 'UNKNOWN CONTACT')) if pd.notna(row.get('vessel_name')) else 'UNKNOWN CONTACT'
            }
            
            # Only include detections with valid coordinates
            if detection['lat'] != 0 and detection['lon'] != 0:
                detections.append(detection)
        
        return detections
    
    except Exception as e:
        print(f"Error loading detections: {e}")
        return []

def find_region_bounds(region_name):
    """Find bounds for a given region name using actual detection data."""
    if not region_name:
        return None
    
    # Load detections to get actual regional data
    detections = load_detections()
    if not detections:
        return get_fallback_bounds(region_name)
    
    # Normalize the region name for matching
    normalized = region_name.lower().strip()
    
    # Find detections that match the region
    matching_detections = []
    
    for detection in detections:
        if 'region' in detection and detection['region']:
            detection_region = detection['region'].lower()
            detection_description = detection.get('area_description', '').lower()
            detection_country = detection.get('country', '').lower()
            
            # Check for matches in region name, description, or country
            if (normalized in detection_region or 
                normalized in detection_description or
                normalized in detection_country or
                any(word in detection_region for word in normalized.split()) or
                any(word in detection_description for word in normalized.split())):
                matching_detections.append(detection)
    
    # If we found matching detections, calculate bounds from them
    if matching_detections:
        lats = [d['lat'] for d in matching_detections if d['lat'] != 0]
        lons = [d['lon'] for d in matching_detections if d['lon'] != 0]
        
        if lats and lons:
            # Add some padding around the detection area
            lat_padding = (max(lats) - min(lats)) * 0.2 + 0.5  # At least 0.5 degrees
            lon_padding = (max(lons) - min(lons)) * 0.2 + 1.0  # At least 1.0 degrees
            
            return {
                'lat_min': max(min(lats) - lat_padding, -90),
                'lon_min': max(min(lons) - lon_padding, -180),
                'lat_max': min(max(lats) + lat_padding, 90),
                'lon_max': min(max(lons) + lon_padding, 180)
            }
    
    # Fallback to predefined regions if no detections match
    return get_fallback_bounds(region_name)

def get_fallback_bounds(region_name):
    """Get fallback bounds from predefined regions."""
    if not region_name:
        return None
        
    normalized = region_name.lower().strip()
    
    # Direct match
    if normalized in ARCTIC_REGIONS:
        bounds = ARCTIC_REGIONS[normalized]
        return {
            'lat_min': bounds[0],
            'lon_min': bounds[1], 
            'lat_max': bounds[2],
            'lon_max': bounds[3]
        }
    
    # Partial match
    for region_key, bounds in ARCTIC_REGIONS.items():
        if normalized in region_key or region_key in normalized:
            return {
                'lat_min': bounds[0],
                'lon_min': bounds[1],
                'lat_max': bounds[2], 
                'lon_max': bounds[3]
            }
    
    # Check for common keywords in your data
    region_keywords = {
        'fram': 'fram strait',
        'greenland': 'greenland sea', 
        'svalbard': 'svalbard',
        'arctic ocean': 'arctic ocean',
        'norwegian': 'norwegian sea',
        'barents': 'barents sea',
        'high arctic': 'arctic',
        'jan mayen': 'greenland sea'
    }
    
    # Check keywords
    for keyword, actual_region in region_keywords.items():
        if keyword in normalized:
            if actual_region in ARCTIC_REGIONS:
                bounds = ARCTIC_REGIONS[actual_region]
                return {
                    'lat_min': bounds[0],
                    'lon_min': bounds[1],
                    'lat_max': bounds[2],
                    'lon_max': bounds[3]
                }
    
    return None

# Enhanced HTML template with confidence-based ship icons
MAP_ONLY_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Arctic Overwatch - Enhanced Ship Detection</title>
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
        .ship-marker {
            display: flex;
            justify-content: center;
            align-items: center;
            border-radius: 50%;
            border: 2px solid rgba(255,255,255,0.3);
            background: rgba(0,0,0,0.7);
            backdrop-filter: blur(2px);
        }
        .confidence-high { color: #28a745; }
        .confidence-medium { color: #ffc107; }
        .confidence-low { color: #dc3545; }
        
        .popup-header {
            background: linear-gradient(135deg, #1a1a1a, #2d3748);
            color: white;
            padding: 10px;
            margin: -10px -10px 10px -10px;
            border-radius: 5px 5px 0 0;
        }
        .popup-section {
            margin: 8px 0;
            padding: 8px;
            background: rgba(0,0,0,0.05);
            border-radius: 4px;
        }
        .confidence-bar {
            width: 100%;
            height: 20px;
            background: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }
        .confidence-fill {
            height: 100%;
            transition: width 0.3s ease;
        }
        .confidence-text {
            font-weight: bold;
            text-align: center;
            line-height: 20px;
            color: white;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.7);
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
            setupMessageListener();
        });

        function initializeMap() {
            // Initialize the map - default to Arctic region
            map = L.map('map').setView([75, -20], 5);
            
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

        function setupMessageListener() {
            window.addEventListener('message', function(event) {
                if (event.data && event.data.type === 'zoomToRegion' && event.data.bounds) {
                    const bounds = event.data.bounds;
                    const leafletBounds = L.latLngBounds(
                        [bounds.lat_min, bounds.lon_min],
                        [bounds.lat_max, bounds.lon_max]
                    );
                    
                    // Zoom to the region with some padding
                    map.fitBounds(leafletBounds, {
                        padding: [20, 20],
                        maxZoom: 20
                    });
                    
                    console.log('Zoomed to region:', bounds);
                }
            });
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

        function getShipIcon(shipType) {
            const iconMap = {
                'cargo ship': 'fa-ship',
                'container ship': 'fa-boxes-stacked',
                'tanker': 'fa-oil-can',
                'fishing vessel': 'fa-fish',
                'research vessel': 'fa-flask',
                'supply vessel': 'fa-truck',
                'cruise ship': 'fa-sailboat',
                'military vessel': 'fa-shield-halved',
                'unknown vessel': 'fa-question'
            };
            
            const type = shipType.toLowerCase();
            for (const [key, icon] of Object.entries(iconMap)) {
                if (type.includes(key)) {
                    return icon;
                }
            }
            return 'fa-ship'; // default
        }

        function getConfidenceColor(confidence) {
            if (confidence >= 75) return '#28a745'; // Green for high confidence
            if (confidence >= 50) return '#ffc107'; // Yellow for medium confidence
            return '#dc3545'; // Red for low confidence
        }

        function getConfidenceClass(confidence) {
            if (confidence >= 75) return 'confidence-high';
            if (confidence >= 50) return 'confidence-medium';
            return 'confidence-low';
        }

        function updateMap() {
            // Clear existing markers
            detectionsLayer.clearLayers();
            
            if (detections.length === 0) return;
            
            // Add markers for each detection
            detections.forEach(detection => {
                if (detection.lat && detection.lon) {
                    const confidence = detection.confidence || 50;
                    const shipIcon = getShipIcon(detection.ship_type || 'unknown vessel');
                    const confidenceColor = getConfidenceColor(confidence);
                    const confidenceClass = getConfidenceClass(confidence);
                    
                    const icon = L.divIcon({
                        className: `ship-marker ${confidenceClass}`,
                        html: `<i class="fas ${shipIcon}" style="color: ${confidenceColor}; font-size: 18px;"></i>`,
                        iconSize: [35, 35],
                        iconAnchor: [17, 17]
                    });
                    
                    const marker = L.marker([detection.lat, detection.lon], {icon})
                        .bindPopup(createEnhancedPopupContent(detection));
                    
                    detectionsLayer.addLayer(marker);
                }
            });
            
            // Fit map to detections if any exist and no specific region zoom has been set
            if (detections.length > 0 && detections.some(d => d.lat && d.lon)) {
                const group = new L.featureGroup(detectionsLayer.getLayers());
                if (map.getZoom() <= 6) {  // Only auto-fit if at default zoom level
                    map.fitBounds(group.getBounds().pad(0.1));
                }
            }
        }

        function createEnhancedPopupContent(detection) {
            const confidence = detection.confidence || 50;
            const confidenceColor = getConfidenceColor(confidence);
            const vesselName = detection.vessel_name || 'UNKNOWN CONTACT';
            const shipType = detection.ship_type || 'Unknown Vessel';
            const shipModel = detection.ship_model || 'Unidentified';
            
            let dimensionsInfo = '';
            if (detection.length_m !== 'Unknown') {
                dimensionsInfo = `
                    <div class="popup-section">
                        <strong>Vessel Dimensions:</strong><br>
                        Length: ${detection.length_m}m | Beam: ${detection.beam_m}m | Draft: ${detection.draft_m}m<br>
                        Tonnage: ${detection.tonnage_gt} GT
                    </div>
                `;
            }
            
            let technicalInfo = '';
            if (detection.engine_type !== 'Unknown') {
                technicalInfo = `
                    <div class="popup-section">
                        <strong>Technical Specifications:</strong><br>
                        Engine: ${detection.engine_type}<br>
                        Model: ${shipModel}<br>
                        Flag State: ${detection.flag_state}
                    </div>
                `;
            }
            
            let locationInfo = '';
            if (detection.region || detection.area_description) {
                locationInfo = `
                    <div class="popup-section">
                        ${detection.region ? `<strong>Region:</strong> ${detection.region}<br>` : ''}
                        ${detection.area_description ? `<strong>Location:</strong> ${detection.area_description}<br>` : ''}
                        ${detection.country ? `<strong>Jurisdiction:</strong> ${detection.country}` : ''}
                    </div>
                `;
            }
            
            const aisInfo = detection.has_ais_match ? 
                '<span style="color: #28a745; font-weight: bold;">✓ AIS Match</span>' : 
                '<span style="color: #dc3545; font-weight: bold;">✗ No AIS Match (Dark Vessel)</span>';
            
            return `
                <div style="min-width: 300px; max-width: 400px;">
                    <div class="popup-header">
                        <h6 style="margin: 0; font-size: 16px;">
                            <i class="fas ${getShipIcon(shipType)}"></i> ${vesselName}
                        </h6>
                        <div style="font-size: 12px; opacity: 0.8;">Detection #${detection.id} | ${shipType}</div>
                    </div>
                    
                    <div class="popup-section">
                        <strong>ML Confidence:</strong>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%; background: ${confidenceColor};">
                                <div class="confidence-text">${confidence.toFixed(1)}%</div>
                            </div>
                        </div>
                    </div>
                    
                    <div class="popup-section">
                        <strong>Position:</strong> ${detection.lat.toFixed(4)}°N, ${Math.abs(detection.lon).toFixed(4)}°${detection.lon < 0 ? 'W' : 'E'}<br>
                        <strong>Radar Signature:</strong> ${detection.intensity_db.toFixed(1)} dB | ${detection.area_pixels} pixels<br>
                        <strong>Status:</strong> ${aisInfo}
                    </div>
                    
                    ${technicalInfo}
                    ${dimensionsInfo}
                    ${locationInfo}
                    
                    ${detection.nearest_mmsi ? `
                        <div class="popup-section">
                            <strong>AIS Correlation:</strong><br>
                            MMSI: ${detection.nearest_mmsi}<br>
                            Distance: ${Math.round(detection.nearest_distance_m)}m
                        </div>
                    ` : ''}
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
    """Serve the enhanced map-only page."""
    return render_template_string(MAP_ONLY_TEMPLATE)

@app.route('/api/detections')
def get_detections():
    """API endpoint to get enhanced detection data."""
    detections = load_detections()
    return jsonify({
        'success': True,
        'detections': detections,
        'count': len(detections)
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_region():
    """Enhanced API endpoint for regional analysis with ship data."""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        
        if not message:
            return jsonify({'error': 'No message provided'}), 400
        
        # Extract region from the analyze command
        region_match = None
        if message.lower().startswith('analyze'):
            # Try to extract region name from various patterns
            import re
            patterns = [
                r'analyze\s+"([^"]+)"',  # "analyze Fram Strait"
                r'analyze\s+([^"]+?)(?:\s+region)?$',  # analyze Fram Strait
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message, re.IGNORECASE)
                if match:
                    region_match = match.group(1).strip()
                    break
        
        if not region_match:
            return jsonify({'error': 'Could not extract region from message'}), 400
        
        # Load detections and analyze the region
        detections = load_detections()
        
        # Find detections in the requested region
        normalized_region = region_match.lower()
        matching_detections = []
        
        for detection in detections:
            region = detection.get('region', '').lower()
            area_desc = detection.get('area_description', '').lower()
            country = detection.get('country', '').lower()
            
            if (normalized_region in region or 
                normalized_region in area_desc or
                normalized_region in country or
                any(word in region for word in normalized_region.split()) or
                any(word in area_desc for word in normalized_region.split())):
                matching_detections.append(detection)
        
        # Calculate statistics
        total_detections = len(matching_detections)
        high_confidence = sum(1 for d in matching_detections if d.get('confidence', 50) >= 75)
        medium_confidence = sum(1 for d in matching_detections if 50 <= d.get('confidence', 50) < 75)
        low_confidence = sum(1 for d in matching_detections if d.get('confidence', 50) < 50)
        dark_vessels = sum(1 for d in matching_detections if not d.get('has_ais_match', False))
        
        # Get ship type distribution
        ship_types = {}
        for detection in matching_detections:
            ship_type = detection.get('ship_type', 'Unknown Vessel')
            ship_types[ship_type] = ship_types.get(ship_type, 0) + 1
        
        # Get top regions
        region_counts = {}
        for detection in detections:
            region = detection.get('region', 'Unknown')
            region_counts[region] = region_counts.get(region, 0) + 1
        
        top_regions = sorted(region_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_regions_formatted = [{'region': r[0], 'count': r[1]} for r in top_regions]
        
        # Find the bounds for the region
        bounds = find_region_bounds(region_match)
        
        # Generate narrative
        confidence_desc = f"High confidence: {high_confidence}, Medium: {medium_confidence}, Low: {low_confidence}"
        
        # Determine threat level
        threat_level = "LOW"
        threat_color = "#28a745"
        if dark_vessels > 3 or (dark_vessels > 1 and total_detections > 5):
            threat_level = "HIGH"
            threat_color = "#dc3545"
        elif dark_vessels > 1 or total_detections > 8:
            threat_level = "MEDIUM" 
            threat_color = "#ffc107"
        
        # Most common ship type
        most_common_ship = max(ship_types.items(), key=lambda x: x[1]) if ship_types else ('Unknown', 0)
        
        narrative = f"""**Regional Analysis: {region_match}**

**Detection Summary:** Found {total_detections} vessel detections in the {region_match} region.

**Threat Level:** {threat_level}

**Confidence Distribution:** {confidence_desc}

**Primary Vessel Type:** {most_common_ship[0]} ({most_common_ship[1]} detections)

**Dark Vessels:** {dark_vessels} vessels detected without AIS correlation, indicating potential untracked maritime activity.

**Assessment:** This region shows {'high' if total_detections > 5 else 'moderate' if total_detections > 2 else 'low'} maritime traffic density with {'significant' if dark_vessels > 2 else 'some' if dark_vessels > 0 else 'no'} dark vessel activity requiring further investigation."""
        
        response_data = {
            'narrative': narrative,
            'requestedRegion': region_match,
            'resolvedRegion': region_match,
            'totalDetections': len(detections),
            'regionalDetections': total_detections,
            'highConfidenceDetections': high_confidence,
            'mediumConfidenceDetections': medium_confidence,
            'lowConfidenceDetections': low_confidence,
            'darkVessels': dark_vessels,
            'threatLevel': threat_level,
            'threatColor': threat_color,
            'shipTypes': ship_types,
            'mostCommonShip': most_common_ship[0],
            'topRegions': top_regions_formatted,
            'bounds': bounds,
            'matchingDetections': matching_detections
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/api/analyze')
def analyze_region_get():
    """API endpoint to get region bounds for analysis (GET method for backward compatibility)."""
    region_name = request.args.get('region', '').strip()
    
    if not region_name:
        return jsonify({
            'success': False,
            'error': 'No region specified'
        }), 400
    
    bounds = find_region_bounds(region_name)
    
    if bounds:
        return jsonify({
            'success': True,
            'region': region_name,
            'bounds': bounds,
            'message': f'Found bounds for {region_name}'
        })
    else:
        # Return a default Arctic bounds if region not found
        default_bounds = {
            'lat_min': 70.0,
            'lon_min': -180.0,
            'lat_max': 90.0,
            'lon_max': 180.0
        }
        return jsonify({
            'success': True,
            'region': region_name,
            'bounds': default_bounds,
            'message': f'Region "{region_name}" not found, showing general Arctic area',
            'fallback': True
        })

@app.route('/api/ship-types')
def get_ship_types():
    """Get ship type distribution across all detections."""
    detections = load_detections()
    
    ship_types = {}
    confidence_by_type = {}
    
    for detection in detections:
        ship_type = detection.get('ship_type', 'Unknown Vessel')
        confidence = detection.get('confidence', 50)
        
        if ship_type not in ship_types:
            ship_types[ship_type] = 0
            confidence_by_type[ship_type] = []
        
        ship_types[ship_type] += 1
        confidence_by_type[ship_type].append(confidence)
    
    # Calculate average confidence by type
    ship_type_data = []
    for ship_type, count in ship_types.items():
        avg_confidence = sum(confidence_by_type[ship_type]) / len(confidence_by_type[ship_type])
        ship_type_data.append({
            'type': ship_type,
            'count': count,
            'avgConfidence': round(avg_confidence, 1)
        })
    
    # Sort by count
    ship_type_data.sort(key=lambda x: x['count'], reverse=True)
    
    return jsonify({
        'success': True,
        'shipTypes': ship_type_data,
        'totalTypes': len(ship_type_data)
    })

@app.route('/api/confidence-stats')
def get_confidence_stats():
    """Get confidence level statistics."""
    detections = load_detections()
    
    high_confidence = [d for d in detections if d.get('confidence', 50) >= 75]
    medium_confidence = [d for d in detections if 50 <= d.get('confidence', 50) < 75]
    low_confidence = [d for d in detections if d.get('confidence', 50) < 50]
    
    return jsonify({
        'total': len(detections),
        'high': {
            'count': len(high_confidence),
            'percentage': round(len(high_confidence) / len(detections) * 100, 1) if detections else 0
        },
        'medium': {
            'count': len(medium_confidence),
            'percentage': round(len(medium_confidence) / len(detections) * 100, 1) if detections else 0
        },
        'low': {
            'count': len(low_confidence),
            'percentage': round(len(low_confidence) / len(detections) * 100, 1) if detections else 0
        }
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
            enable_ml = request.form.get('enable_ml', 'true').lower() == 'true'
            
            # Save uploaded file
            filename = secure_filename(file.filename)
            unique_filename = f"{uuid.uuid4()}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Process the file with enhanced detection
            success = process_sar_image_enhanced(filepath, sigma, window, min_area_pixels, enable_ml)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if success:
                # Reload detections after processing
                detections = load_detections()
                return jsonify({
                    'success': True,
                    'message': 'File processed successfully with enhanced ship detection',
                    'detections': detections,
                    'count': len(detections)
                })
            else:
                return jsonify({'success': False, 'error': 'Enhanced processing failed'}), 500
            
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
    else:
        return jsonify({'success': False, 'error': 'Invalid file type'}), 400

def process_sar_image_enhanced(filepath, sigma, window, min_area_pixels, enable_ml=True):
    """
    Process SAR image with enhanced ship detection capabilities.
    Replace this with your actual enhanced SAR processing logic.
    """
    try:
        print(f"Processing SAR image with enhanced detection: {filepath}")
        print(f"Parameters: sigma={sigma}, window={window}, min_area={min_area_pixels}, ML={enable_ml}")
        
        # Placeholder for enhanced SAR processing
        # Your actual implementation would:
        # 1. Load the SAR image from filepath
        # 2. Apply your detection algorithm with the given parameters
        # 3. Run ML classification for ship types and confidence scoring
        # 4. Perform vessel identification and specification lookup
        # 5. Save enhanced results to myoutput/detections.csv with all new columns
        # 6. Return True if successful
        
        # Example integration point:
        # from sar_app.enhanced_detection import process_image_with_ml
        # success = process_image_with_ml(filepath, sigma, window, min_area_pixels, enable_ml)
        
        return True  # Placeholder return
        
    except Exception as e:
        print(f"Error processing SAR image: {e}")
        return False

@app.route('/api/regions')
def get_regions():
    """Get available regions from detection data with enhanced statistics."""
    detections = load_detections()
    
    regions = {}
    for detection in detections:
        region = detection.get('region', '')
        area_desc = detection.get('area_description', '')
        country = detection.get('country', '')
        confidence = detection.get('confidence', 50)
        ship_type = detection.get('ship_type', 'Unknown Vessel')
        has_ais = detection.get('has_ais_match', False)
        
        if region and region not in regions:
            regions[region] = {
                'name': region,
                'count': 0,
                'highConfidenceCount': 0,
                'darkVesselCount': 0,
                'avgConfidence': 0,
                'confidenceSum': 0,
                'sample_description': area_desc,
                'country': country,
                'shipTypes': {}
            }
        
        if region:
            regions[region]['count'] += 1
            regions[region]['confidenceSum'] += confidence
            
            if confidence >= 75:
                regions[region]['highConfidenceCount'] += 1
            
            if not has_ais:
                regions[region]['darkVesselCount'] += 1
            
            # Track ship types in this region
            if ship_type not in regions[region]['shipTypes']:
                regions[region]['shipTypes'][ship_type] = 0
            regions[region]['shipTypes'][ship_type] += 1
    
    # Calculate averages and format data
    for region_data in regions.values():
        region_data['avgConfidence'] = round(region_data['confidenceSum'] / region_data['count'], 1)
        del region_data['confidenceSum']  # Remove intermediate calculation
        
        # Get most common ship type
        if region_data['shipTypes']:
            most_common = max(region_data['shipTypes'].items(), key=lambda x: x[1])
            region_data['primaryShipType'] = most_common[0]
        else:
            region_data['primaryShipType'] = 'Unknown'
    
    # Sort by detection count
    sorted_regions = sorted(regions.values(), key=lambda x: x['count'], reverse=True)
    
    return jsonify({
        'success': True,
        'regions': sorted_regions,
        'total_regions': len(sorted_regions)
    })

@app.route('/api/stats')
def get_stats():
    """Get enhanced detection statistics."""
    detections = load_detections()
    
    if not detections:
        return jsonify({
            'total_detections': 0,
            'ais_matches': 0,
            'dark_vessels': 0,
            'avg_intensity': 0,
            'avg_confidence': 0,
            'ship_type_diversity': 0
        })
    
    total_detections = len(detections)
    ais_matches = sum(1 for d in detections if d['has_ais_match'])
    dark_vessels = total_detections - ais_matches
    avg_intensity = sum(d['intensity_db'] for d in detections) / total_detections
    avg_confidence = sum(d.get('confidence', 50) for d in detections) / total_detections
    
    # Count unique ship types
    ship_types = set(d.get('ship_type', 'Unknown') for d in detections)
    ship_type_diversity = len(ship_types)
    
    # Confidence distribution
    high_conf = sum(1 for d in detections if d.get('confidence', 50) >= 75)
    medium_conf = sum(1 for d in detections if 50 <= d.get('confidence', 50) < 75)
    low_conf = sum(1 for d in detections if d.get('confidence', 50) < 50)
    
    return jsonify({
        'total_detections': total_detections,
        'ais_matches': ais_matches,
        'dark_vessels': dark_vessels,
        'avg_intensity': round(avg_intensity, 1),
        'avg_confidence': round(avg_confidence, 1),
        'ship_type_diversity': ship_type_diversity,
        'confidence_distribution': {
            'high': high_conf,
            'medium': medium_conf,
            'low': low_conf
        }
    })

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'version': '2.0-enhanced',
        'features': [
            'ML Confidence Scoring',
            'Ship Type Classification',
            'Vessel Identification',
            'Technical Specifications',
            'Enhanced Regional Analysis'
        ]
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)