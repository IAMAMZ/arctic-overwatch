# SAR Vessel Detection Web Application

A modern web application for visualizing SAR (Synthetic Aperture Radar) vessel detections and cross-referencing them with AIS (Automatic Identification System) data.

## Features

- 🗺️ **Interactive Map**: View vessel detections on an interactive map with multiple base layers
- 🚢 **Vessel Detection**: Process Sentinel-1 SAR images to detect bright targets (potential vessels)
- 📡 **AIS Integration**: Cross-reference detections with AIS data to identify known vessels
- 🕵️ **Dark Vessel Detection**: Identify potential dark vessels (ships not transmitting AIS)
- 📊 **Statistics Dashboard**: Real-time statistics on detections and AIS matches
- 📤 **File Upload**: Upload and process new SAR images through the web interface
- 📱 **Responsive Design**: Modern, responsive UI that works on all devices

## Quick Start

### Option 1: Using the startup script (Recommended)
```bash
python run_webapp.py
```

### Option 2: Manual start
```bash
# Install dependencies
pip install -r requirements.txt

# Start the web application
python app.py
```

Then open your browser to: **http://localhost:5000**

## Processing SAR Images

### Through the Web Interface
1. Open the web application in your browser
2. Use the upload panel on the right side
3. Click or drag-and-drop your SAR image (.tif/.tiff format)
4. Adjust detection parameters if needed:
   - **Sigma**: Threshold multiplier (default: 3.0)
   - **Window**: Analysis window size (default: 15)
   - **Min Area**: Minimum detection area in pixels (default: 3)
5. Click upload to process

### Through Command Line
You can also process images using the original CLI tool:
```bash
python -m sar_app.cli --image "your_image.tif" --out "myoutput"
```

## Understanding the Interface

### Map Features
- **Green Ships**: Detections with AIS matches (known vessels)
- **Red Ships**: Detections without AIS matches (potential dark vessels)
- **Yellow Spies**: High-intensity detections without AIS (suspicious)
- Click on any marker to see detection details
- Use layer control to switch between satellite, street, and dark map views

### Detection List
- Shows all detections with key statistics
- Click any detection to focus on it on the map
- Green border: AIS match found
- Red border: No AIS match (potential dark vessel)

### Statistics Panel
- **Total Detections**: Number of vessels detected in the SAR image
- **AIS Matches**: Detections that match with AIS transponder data
- **Potential Dark Vessels**: Detections without AIS matches
- **Average Intensity**: Mean radar intensity of detections in dB

## AIS Data Integration

The application supports multiple AIS data sources:

1. **AIS Stream API** - Set environment variable `AISSTREAM_API_KEY`
2. **Barents Watch** - Free for Arctic regions (Norway)
3. **Simulated Data** - For demonstration purposes

### Setting up Real AIS Data

To use real AIS data, you can:

1. **AIS Stream**: Register at [aisstream.io](https://aisstream.io) and set your API key:
   ```bash
   export AISSTREAM_API_KEY="your_api_key_here"
   ```

2. **Barents Watch**: Automatically used for Arctic coordinates (>65°N)

3. **Custom AIS CSV**: Use the original CLI with your own AIS data:
   ```bash
   python -m sar_app.cli --image "image.tif" --ais "your_ais.csv" --out "myoutput"
   ```

## Technical Details

### Supported File Formats
- **Input**: GeoTIFF (.tif, .tiff) Sentinel-1 SAR images
- **Output**: CSV tables and GeoJSON for mapping

### Detection Algorithm
- Uses adaptive thresholding based on local statistics
- Applies morphological operations to reduce noise
- Calculates center of mass for precise positioning
- Converts pixel coordinates to geographic coordinates (WGS84)

### AIS Matching
- Spatial matching within configurable distance threshold (default: 3km)
- Temporal matching within time window (default: ±60 minutes)
- Haversine distance calculation for accurate geographic matching

## Architecture

- **Backend**: Flask web server with REST API endpoints
- **Frontend**: Modern HTML5/CSS3/JavaScript with Bootstrap 5
- **Mapping**: Leaflet.js for interactive maps
- **Detection**: Python with NumPy, SciPy, and Rasterio
- **Geospatial**: PyProj for coordinate transformations

## API Endpoints

- `GET /` - Main web application
- `GET /api/detections` - Get current detections with AIS matching
- `GET /api/geojson` - Get detections in GeoJSON format
- `GET /api/stats` - Get detection statistics
- `POST /api/upload` - Upload and process new SAR images

## Troubleshooting

### No detections showing?
- Make sure you have processed a SAR image first
- Check that `myoutput/detections.csv` exists
- Verify the SAR image has valid geographic coordinates

### Upload not working?
- Ensure uploaded files are in TIFF format
- Check file size (large SAR images may take time to process)
- Look at browser console for error messages

### Map not loading?
- Check internet connection (map tiles are loaded from online sources)
- Try refreshing the page
- Verify JavaScript is enabled in your browser

## Development

To modify or extend the application:

1. **Backend changes**: Edit `app.py` for new API endpoints or processing logic
2. **Frontend changes**: Edit `templates/index.html` and `static/js/app.js`
3. **Styling changes**: Add custom CSS to `static/css/` or modify inline styles
4. **Detection algorithm**: Modify `sar_app/detector.py` for different detection methods

## License

This project builds on the existing SAR detection framework and adds a modern web interface for visualization and analysis.
