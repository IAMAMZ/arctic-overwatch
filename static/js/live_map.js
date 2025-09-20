// Arctic Overwatch Live Map System
class ArcticLiveMap {
    constructor() {
        this.map = null;
        this.detectionsLayer = null;
        this.historicalData = [];
        this.currentTimeIndex = 0;
        this.isPlaying = false;
        this.playbackSpeed = 2; // hours per second
        this.playbackInterval = null;
        this.currentDetections = [];
        this.filterType = 'all';
        
        // Time tracking
        this.startDate = new Date('2023-01-01T00:00:00Z');
        this.endDate = new Date('2023-12-31T23:59:59Z');
        this.currentDate = new Date(this.startDate);
        
        this.init();
    }

    async init() {
        await this.loadHistoricalData();
        this.initializeMap();
        this.setupEventListeners();
        this.updateDisplay();
    }

    async loadHistoricalData() {
        try {
            const response = await fetch('/api/historical-data');
            this.historicalData = await response.json();
            console.log(`Loaded ${this.historicalData.length} historical detections`);
        } catch (error) {
            console.error('Error loading historical data:', error);
            // Generate some sample data if API fails
            this.generateSampleData();
        }
    }

    generateSampleData() {
        // Generate sample data for demo
        this.historicalData = [];
        const startDate = new Date('2023-01-01T00:00:00Z');
        
        for (let i = 0; i < 1000; i++) {
            const randomDate = new Date(startDate.getTime() + Math.random() * (this.endDate.getTime() - startDate.getTime()));
            this.historicalData.push({
                id: i,
                timestamp: randomDate.toISOString(),
                lat: 75 + Math.random() * 10, // Arctic region
                lon: -60 + Math.random() * 120,
                vessel_type: ['Cargo', 'Tanker', 'Fishing', 'Research', 'Unknown'][Math.floor(Math.random() * 5)],
                has_ais_match: Math.random() > 0.3,
                intensity_db: 20 + Math.random() * 15,
                area_pixels: 5 + Math.random() * 20,
                mmsi: Math.random() > 0.3 ? Math.floor(Math.random() * 900000000 + 100000000).toString() : '',
                sog: 5 + Math.random() * 15,
                cog: Math.random() * 360,
                length: 50 + Math.random() * 250,
                weather_conditions: ['Clear', 'Fog', 'Snow', 'Storm'][Math.floor(Math.random() * 4)],
                ice_conditions: ['Open Water', 'Light Ice', 'Heavy Ice', 'Pack Ice'][Math.floor(Math.random() * 4)]
            });
        }
        
        // Sort by timestamp
        this.historicalData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    }

    initializeMap() {
        // Initialize the map focused on Arctic region
        this.map = L.map('map').setView([80, -20], 4);
        
        // Add dark theme tile layer
        const darkLayer = L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors © CARTO'
        });
        
        const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
            attribution: 'Tiles © Esri — Source: Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community'
        });
        
        // Add default layer
        darkLayer.addTo(this.map);
        
        // Layer control
        const baseLayers = {
            "Dark Theme": darkLayer,
            "Satellite": satelliteLayer
        };
        
        L.control.layers(baseLayers).addTo(this.map);
        
        // Initialize detections layer
        this.detectionsLayer = L.layerGroup().addTo(this.map);
        
        // Add scale control
        L.control.scale({position: 'bottomright'}).addTo(this.map);
    }

    setupEventListeners() {
        // Play/Pause button
        document.getElementById('play-pause-btn').addEventListener('click', () => {
            this.togglePlayback();
        });
        
        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => {
            this.resetPlayback();
        });
        
        // Speed control
        document.getElementById('speed-select').addEventListener('change', (e) => {
            this.playbackSpeed = parseInt(e.target.value);
            if (this.isPlaying) {
                this.stopPlayback();
                this.startPlayback();
            }
        });
        
        // Filter control
        document.getElementById('filter-select').addEventListener('change', (e) => {
            this.filterType = e.target.value;
            this.updateMap();
        });
        
        // Timeline click
        document.querySelector('.timeline').addEventListener('click', (e) => {
            const rect = e.currentTarget.getBoundingClientRect();
            const clickX = e.clientX - rect.left;
            const progress = clickX / rect.width;
            this.seekToProgress(progress);
        });
    }

    togglePlayback() {
        if (this.isPlaying) {
            this.stopPlayback();
        } else {
            this.startPlayback();
        }
    }

    startPlayback() {
        this.isPlaying = true;
        document.getElementById('play-pause-btn').innerHTML = '<i class="fas fa-pause"></i>';
        
        this.playbackInterval = setInterval(() => {
            this.advanceTime();
        }, 1000 / this.playbackSpeed);
    }

    stopPlayback() {
        this.isPlaying = false;
        document.getElementById('play-pause-btn').innerHTML = '<i class="fas fa-play"></i>';
        
        if (this.playbackInterval) {
            clearInterval(this.playbackInterval);
            this.playbackInterval = null;
        }
    }

    resetPlayback() {
        this.stopPlayback();
        this.currentTimeIndex = 0;
        this.currentDate = new Date(this.startDate);
        this.currentDetections = [];
        this.updateDisplay();
    }

    advanceTime() {
        // Advance time by 1 hour
        this.currentDate.setHours(this.currentDate.getHours() + 1);
        
        // Check if we've reached the end
        if (this.currentDate >= this.endDate) {
            this.stopPlayback();
            return;
        }
        
        // Update detections for current time
        this.updateDetectionsForCurrentTime();
        this.updateDisplay();
    }

    updateDetectionsForCurrentTime() {
        const currentTime = this.currentDate.getTime();
        const oneHourAgo = currentTime - (60 * 60 * 1000);
        
        // Get detections from the last hour
        this.currentDetections = this.historicalData.filter(detection => {
            const detectionTime = new Date(detection.timestamp).getTime();
            return detectionTime >= oneHourAgo && detectionTime <= currentTime;
        });
        
        // Apply filter
        if (this.filterType !== 'all') {
            this.currentDetections = this.currentDetections.filter(detection => {
                switch (this.filterType) {
                    case 'matched':
                        return detection.has_ais_match;
                    case 'unmatched':
                        return !detection.has_ais_match;
                    case 'dark':
                        return !detection.has_ais_match && detection.vessel_type === 'Unknown';
                    default:
                        return true;
                }
            });
        }
    }

    updateDisplay() {
        this.updateTimeDisplay();
        this.updateMap();
        this.updateStats();
        this.updateDetectionList();
        this.updateTimeline();
    }

    updateTimeDisplay() {
        const timeStr = this.currentDate.toISOString().replace('T', ' ').substring(0, 19);
        document.getElementById('time-display').textContent = timeStr;
        document.getElementById('current-time').textContent = timeStr;
    }

    updateMap() {
        // Clear existing markers
        this.detectionsLayer.clearLayers();
        
        if (this.currentDetections.length === 0) return;
        
        // Create custom icons
        const matchedIcon = L.divIcon({
            className: 'vessel-marker matched',
            html: '<div class="vessel-icon matched"></div>',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });
        
        const unmatchedIcon = L.divIcon({
            className: 'vessel-marker unmatched',
            html: '<div class="vessel-icon unmatched"></div>',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });
        
        const darkVesselIcon = L.divIcon({
            className: 'vessel-marker dark-vessel',
            html: '<div class="vessel-icon dark-vessel"></div>',
            iconSize: [20, 20],
            iconAnchor: [10, 10]
        });
        
        // Add markers for each detection
        this.currentDetections.forEach(detection => {
            if (detection.lat && detection.lon) {
                let icon;
                if (detection.has_ais_match) {
                    icon = matchedIcon;
                } else if (detection.vessel_type === 'Unknown') {
                    icon = darkVesselIcon;
                } else {
                    icon = unmatchedIcon;
                }
                
                const marker = L.marker([detection.lat, detection.lon], {icon})
                    .bindPopup(this.createPopupContent(detection))
                    .on('click', () => this.showDetectionModal(detection));
                
                this.detectionsLayer.addLayer(marker);
            }
        });
        
        // Fit map to detections if any exist
        if (this.currentDetections.length > 0 && this.currentDetections.some(d => d.lat && d.lon)) {
            const group = new L.featureGroup(this.detectionsLayer.getLayers());
            this.map.fitBounds(group.getBounds().pad(0.1));
        }
    }

    createPopupContent(detection) {
        const status = detection.has_ais_match ? 
            '<span class="badge bg-success">AIS Match</span>' : 
            '<span class="badge bg-danger">No AIS Match</span>';
        
        return `
            <div style="min-width: 200px; color: #000;">
                <h6><i class="fas fa-ship"></i> ${detection.vessel_type} Vessel</h6>
                <p><strong>Position:</strong> ${detection.lat.toFixed(4)}°N, ${Math.abs(detection.lon).toFixed(4)}°W</p>
                <p><strong>Intensity:</strong> ${detection.intensity_db.toFixed(1)} dB</p>
                <p><strong>Status:</strong> ${status}</p>
                <p><strong>Time:</strong> ${new Date(detection.timestamp).toLocaleString()}</p>
                ${detection.mmsi ? `<p><strong>MMSI:</strong> ${detection.mmsi}</p>` : ''}
            </div>
        `;
    }

    updateStats() {
        const totalDetections = this.currentDetections.length;
        const aisMatches = this.currentDetections.filter(d => d.has_ais_match).length;
        const darkVessels = this.currentDetections.filter(d => !d.has_ais_match && d.vessel_type === 'Unknown').length;
        const activeVessels = this.currentDetections.length;
        
        document.getElementById('total-detections').textContent = totalDetections;
        document.getElementById('ais-matches').textContent = aisMatches;
        document.getElementById('dark-vessels').textContent = darkVessels;
        document.getElementById('active-vessels').textContent = activeVessels;
    }

    updateDetectionList() {
        const listContainer = document.getElementById('detection-list');
        
        if (this.currentDetections.length === 0) {
            listContainer.innerHTML = '<p class="text-center text-muted">No active detections</p>';
            return;
        }
        
        // Sort by timestamp (newest first)
        const sortedDetections = [...this.currentDetections].sort((a, b) => 
            new Date(b.timestamp) - new Date(a.timestamp)
        );
        
        const listHTML = sortedDetections.slice(0, 20).map(detection => {
            const statusClass = detection.has_ais_match ? 'matched' : 
                              (detection.vessel_type === 'Unknown' ? 'dark-vessel' : 'unmatched');
            const statusText = detection.has_ais_match ? 'AIS Match' : 
                              (detection.vessel_type === 'Unknown' ? 'Dark Vessel' : 'No AIS Match');
            
            return `
                <div class="detection-card ${statusClass}" onclick="liveMap.showDetectionModal(${JSON.stringify(detection).replace(/"/g, '&quot;')})">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">${detection.vessel_type} Vessel</h6>
                            <small class="text-muted">
                                ${detection.lat.toFixed(4)}°N, ${Math.abs(detection.lon).toFixed(4)}°W
                            </small>
                        </div>
                        <div class="text-end">
                            <div class="vessel-icon ${statusClass}"></div>
                            <br><small>${statusText}</small>
                        </div>
                    </div>
                    <div class="mt-2">
                        <small>
                            <strong>Intensity:</strong> ${detection.intensity_db.toFixed(1)} dB |
                            <strong>Time:</strong> ${new Date(detection.timestamp).toLocaleTimeString()}
                            ${detection.mmsi ? `| <strong>MMSI:</strong> ${detection.mmsi}` : ''}
                        </small>
                    </div>
                </div>
            `;
        }).join('');
        
        listContainer.innerHTML = listHTML;
    }

    updateTimeline() {
        const totalTime = this.endDate.getTime() - this.startDate.getTime();
        const currentTime = this.currentDate.getTime() - this.startDate.getTime();
        const progress = (currentTime / totalTime) * 100;
        
        document.getElementById('timeline-progress').style.width = `${progress}%`;
    }

    seekToProgress(progress) {
        const totalTime = this.endDate.getTime() - this.startDate.getTime();
        const targetTime = this.startDate.getTime() + (totalTime * progress);
        this.currentDate = new Date(targetTime);
        
        this.updateDetectionsForCurrentTime();
        this.updateDisplay();
    }

    showDetectionModal(detection) {
        const modal = new bootstrap.Modal(document.getElementById('detectionModal'));
        const modalBody = document.getElementById('modal-body');
        
        const aisInfo = detection.has_ais_match ? `
            <div class="alert alert-success">
                <h6><i class="fas fa-link"></i> AIS Match Found</h6>
                <p><strong>MMSI:</strong> ${detection.mmsi}</p>
                <p><strong>Vessel Type:</strong> ${detection.vessel_type}</p>
                <p><strong>Speed:</strong> ${detection.sog.toFixed(1)} knots</p>
                <p><strong>Course:</strong> ${detection.cog.toFixed(1)}°</p>
            </div>
        ` : `
            <div class="alert alert-warning">
                <h6><i class="fas fa-exclamation-triangle"></i> No AIS Match</h6>
                <p>This detection has no corresponding AIS signal, which may indicate:</p>
                <ul>
                    <li>A vessel not transmitting AIS (dark vessel)</li>
                    <li>AIS equipment malfunction</li>
                    <li>Vessel in AIS coverage gap</li>
                    <li>False positive detection</li>
                </ul>
            </div>
        `;
        
        modalBody.innerHTML = `
            <div>
                <h6>${detection.vessel_type} Vessel Detection</h6>
                
                <div class="row mb-3">
                    <div class="col-6">
                        <strong>Position:</strong><br>
                        ${detection.lat.toFixed(6)}°N<br>
                        ${Math.abs(detection.lon).toFixed(6)}°W
                    </div>
                    <div class="col-6">
                        <strong>Detection Time:</strong><br>
                        ${new Date(detection.timestamp).toLocaleString()}
                    </div>
                </div>
                
                <div class="row mb-3">
                    <div class="col-6">
                        <strong>Intensity:</strong><br>
                        ${detection.intensity_db.toFixed(2)} dB
                    </div>
                    <div class="col-6">
                        <strong>Area:</strong><br>
                        ${detection.area_pixels} pixels
                    </div>
                </div>
                
                <div class="row mb-3">
                    <div class="col-6">
                        <strong>Weather:</strong><br>
                        ${detection.weather_conditions}
                    </div>
                    <div class="col-6">
                        <strong>Ice Conditions:</strong><br>
                        ${detection.ice_conditions}
                    </div>
                </div>
                
                ${aisInfo}
            </div>
        `;
        
        modal.show();
    }
}

// Initialize the live map when the page loads
let liveMap;
document.addEventListener('DOMContentLoaded', function() {
    liveMap = new ArcticLiveMap();
});
