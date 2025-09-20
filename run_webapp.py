#!/usr/bin/env python3
"""
Startup script for the SAR Vessel Detection Web Application

This script ensures all dependencies are installed and starts the Flask web app.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install required packages if not already installed."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        sys.exit(1)

def check_data_files():
    """Check if detection data files exist."""
    data_paths = [
        "myoutput/detections.csv",
        "myoutput/detections.geojson"
    ]
    
    missing_files = []
    for path in data_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    if missing_files:
        print("Warning: The following data files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nTo generate detection data, run:")
        print("  python -m sar_app.cli --image <your_sar_image.tif> --out myoutput")
        print("\nThe web app will still work but won't show any detections until data is available.")
    else:
        print("✓ Detection data files found")

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["uploads", "static/css", "static/js", "templates"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure verified")

def main():
    """Main startup routine."""
    print("🚀 Starting SAR Vessel Detection Web Application")
    print("=" * 50)
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python version: {sys.version}")
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Check data files
    check_data_files()
    
    print("\n" + "=" * 50)
    print("🌐 Starting web server...")
    print("📍 Open your browser to: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask application
    try:
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
