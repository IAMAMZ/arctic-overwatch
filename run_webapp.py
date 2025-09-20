#!/usr/bin/env python3
"""
Enhanced startup script for the SAR Vessel Detection Backend with Ship Intelligence

This script starts the Flask backend with enhanced ship detection capabilities
including confidence ratings, ship types, and detailed vessel information.
"""

import sys
import subprocess
import os
from pathlib import Path

def install_requirements():
    """Install required packages if not already installed."""
    requirements = [
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
        "pandas>=1.3.0",
        "werkzeug>=2.0.0"
    ]
    
    try:
        for req in requirements:
            subprocess.check_call([sys.executable, "-m", "pip", "install", req], 
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("✓ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        print("Please install manually: pip install flask flask-cors pandas werkzeug")
        sys.exit(1)

def check_data_files():
    """Check if detection data files exist and validate enhanced format."""
    data_paths = [
        "myoutput/detections.csv",
        "myoutput/detections.geojson"
    ]
    
    missing_files = []
    for path in data_paths:
        if not os.path.exists(path):
            missing_files.append(path)
    
    # Check if CSV has enhanced columns
    csv_path = "myoutput/detections.csv"
    if os.path.exists(csv_path):
        try:
            import pandas as pd
            df = pd.read_csv(csv_path, nrows=1)  # Just check headers
            
            required_columns = ['confidence', 'ship_type', 'vessel_name', 'engine_type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️  CSV file exists but missing enhanced columns: {', '.join(missing_columns)}")
                print("   The system will work with basic detection data")
            else:
                print("✓ Enhanced detection data format detected")
                
        except Exception as e:
            print(f"Warning: Could not validate CSV format: {e}")
    
    if missing_files:
        print("Warning: The following data files are missing:")
        for file in missing_files:
            print(f"  - {file}")
        print("\nTo generate enhanced detection data, ensure your processing pipeline includes:")
        print("  - ML confidence scores (0-100%)")
        print("  - Ship type classification")
        print("  - Vessel identification")
        print("  - Engine type detection")
        print("  - Technical specifications")
        print("\nThe backend will serve empty data until files are available.")
    else:
        print("✓ Detection data files found")

def create_directories():
    """Create necessary directories if they don't exist."""
    directories = ["uploads", "myoutput", "static", "templates"]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✓ Directory structure verified")

def create_sample_data():
    """Create sample enhanced data if no data exists."""
    csv_path = "myoutput/detections.csv"
    
    if not os.path.exists(csv_path):
        print("Creating sample enhanced detection data...")
        
        sample_data = """id,row,col,area_pixels,intensity_db,x,y,lon,lat,has_ais_match,nearest_mmsi,nearest_distance_m,nearest_time_delta_s,ais_timestamp,region,area_description,country,confidence,ship_type,ship_model,engine_type,length_m,beam_m,draft_m,tonnage_gt,flag_state,vessel_name
0,3732.000836692215,9042.988898238003,9,26.35483741760254,385138.10402097605,-849555.978282722,-20.613289844055206,81.40475810866143,False,,,,,High Arctic,North of Northeast Greenland in the Fram Strait,International Waters (near Greenland),85.4,Cargo Ship,Bulk Carrier,Marine Diesel,180.5,28.2,12.1,15420,Liberia,ARCTIC PIONEER
1,4750.9904435248145,8688.004081306612,9,25.92176628112793,370938.7113437204,-890315.5625560259,-22.38156659946258,81.113579321309,False,,,,,High Arctic,East of Greenland approaching Fram Strait,International Waters (near Greenland),72.8,Fishing Vessel,Trawler,Marine Diesel,65.8,12.4,5.2,1850,Norway,NORTHERN STAR
2,4959.003215402306,8784.996626414095,9,27.259117126464844,374818.4131480197,-898636.0734311256,-22.359129204051573,81.02939575774161,False,,,,,High Arctic,North Greenland Sea off Northeast Greenland,International Waters (near Greenland),91.2,Container Ship,Feeder Vessel,Marine Gas Turbine,220.4,32.8,13.5,28500,Panama,GREENLAND EXPRESS"""
        
        try:
            with open(csv_path, 'w') as f:
                f.write(sample_data)
            print("✓ Sample enhanced detection data created")
        except Exception as e:
            print(f"Could not create sample data: {e}")

def display_startup_info():
    """Display comprehensive startup information."""
    print("\n" + "=" * 60)
    print("🚢 ARCTIC OVERWATCH - Enhanced Ship Detection System")
    print("=" * 60)
    print("🎯 Features:")
    print("  • Real-time ship detection with ML confidence scoring")
    print("  • Ship type classification (Cargo, Fishing, Military, etc.)")
    print("  • Vessel identification and technical specifications")
    print("  • Confidence-based color coding (Red/Yellow/Green)")
    print("  • Enhanced popup information with ship details")
    print("  • Dark vessel detection (no AIS correlation)")
    print("  • Interactive regional analysis")
    print("  • Intelligent chat interface with ship queries")
    print("\n🌐 Access Points:")
    print("  📍 Flask Backend API: http://localhost:5000")
    print("  📍 Enhanced Map View: http://localhost:5000")
    print("  📍 Next.js Frontend: http://localhost:3000")
    print("\n💡 Try these queries in the chat:")
    print('  • "analyze ships in Fram Strait"')
    print('  • "show me high confidence detections"')
    print('  • "what military vessels are detected?"')
    print('  • "analyze regional shipping patterns"')
    print("\n🛑 Press Ctrl+C to stop the server")
    print("=" * 60)

def main():
    """Enhanced startup routine."""
    print("🚀 Initializing Enhanced Arctic Ship Detection System")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    print(f"✓ Python version: {sys.version}")
    
    # Create directories
    create_directories()
    
    # Install requirements
    install_requirements()
    
    # Check/create data files
    check_data_files()
    create_sample_data()
    
    # Display startup information
    display_startup_info()
    
    # Start the Flask application
    try:
        # Import the enhanced app
        print("\n🔄 Starting enhanced backend server...")
        from app import app
        app.run(debug=True, host='0.0.0.0', port=5000)
    except KeyboardInterrupt:
        print("\n👋 Arctic Overwatch system stopped")
    except ImportError:
        print("\n❌ Could not import app.py. Make sure the enhanced Flask app is saved as 'app.py'")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()