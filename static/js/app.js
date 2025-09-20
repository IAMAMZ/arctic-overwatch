// Global variables
let map;
let detectionsLayer;
let detections = [];
let currentStats = {};

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    initializeMap();
    loadDetections();
    setupEventListeners();
});

function initializeMap() {
    // Initialize the map - default to Arctic region
    map = L.map('map').setView([81, -20], 6);
    
    // Add multiple tile layers
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors'
    });
    
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles © Esri — Source: Esri, DigitalGlobe, GeoEye, Earthstar Geographics, CNES/Airbus DS, USDA, USGS, AeroGRID, IGN, and the GIS User Community'
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

function setupEventListeners() {
    // File upload
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    
    uploadArea.addEventListener('click', () => fileInput.click());
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);
    
    // Refresh button (if needed)
    document.addEventListener('keydown', function(e) {
        if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
            e.preventDefault();
            loadDetections();
        }
    });
}

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.style.backgroundColor = '#f8f9ff';
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.style.backgroundColor = '';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        uploadFile(files[0]);
    }
}

function handleFileSelect(e) {
    if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
    }
}

function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('sigma', document.getElementById('sigma-input').value);
    formData.append('window', document.getElementById('window-input').value);
    formData.append('min_area_pixels', document.getElementById('area-input').value);
    
    const loadingDiv = document.getElementById('upload-loading');
    loadingDiv.style.display = 'block';
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        loadingDiv.style.display = 'none';
        
        if (data.success) {
            // Update detections with new data
            detections = data.detections;
            updateMap();
            updateStats();
            updateDetectionList();
            
            // Show success message
            showNotification('File processed successfully!', 'success');
        } else {
            showNotification(data.error || 'Upload failed', 'error');
        }
    })
    .catch(error => {
        loadingDiv.style.display = 'none';
        showNotification('Upload failed: ' + error.message, 'error');
    });
}

function loadDetections() {
    fetch('/api/detections')
        .then(response => response.json())
        .then(data => {
            if (data.detections) {
                detections = data.detections;
                updateMap();
                updateStats();
                updateDetectionList();
            }
        })
        .catch(error => {
            console.error('Error loading detections:', error);
            showNotification('Failed to load detections', 'error');
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
                // Consider it a potential dark vessel if no AIS match
                icon = detection.intensity_db > 25 ? darkVesselIcon : unmatchedIcon;
            }
            
            const marker = L.marker([detection.lat, detection.lon], {icon})
                .bindPopup(createPopupContent(detection))
                .on('click', () => showDetectionModal(detection));
            
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
        `<span class="badge bg-success">AIS Match</span>` : 
        `<span class="badge bg-danger">No AIS Match</span>`;
    
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
            <button class="btn btn-sm btn-primary" onclick="showDetectionModal(${JSON.stringify(detection).replace(/"/g, '&quot;')})">
                View Details
            </button>
        </div>
    `;
}

function updateStats() {
    const totalDetections = detections.length;
    const aisMatches = detections.filter(d => d.has_ais_match).length;
    const darkVessels = totalDetections - aisMatches;
    const avgIntensity = detections.length > 0 ? 
        detections.reduce((sum, d) => sum + d.intensity_db, 0) / detections.length : 0;
    
    document.getElementById('total-detections').textContent = totalDetections;
    document.getElementById('ais-matches').textContent = aisMatches;
    document.getElementById('dark-vessels').textContent = darkVessels;
    document.getElementById('avg-intensity').textContent = avgIntensity.toFixed(1);
    
    currentStats = {
        totalDetections,
        aisMatches,
        darkVessels,
        avgIntensity
    };
}

function updateDetectionList() {
    const listContainer = document.getElementById('detection-list');
    
    if (detections.length === 0) {
        listContainer.innerHTML = '<p class="text-center text-muted">No detections found</p>';
        return;
    }
    
    const listHTML = detections.map(detection => {
        const statusClass = detection.has_ais_match ? 'matched' : 'unmatched';
        const statusText = detection.has_ais_match ? 'AIS Match' : 'No AIS Match';
        const statusIcon = detection.has_ais_match ? 'fa-check-circle' : 'fa-exclamation-circle';
        
        return `
            <div class="card detection-item ${statusClass} mb-2" onclick="focusOnDetection(${detection.id})">
                <div class="card-body p-3">
                    <div class="d-flex justify-content-between align-items-center">
                        <div>
                            <h6 class="mb-1">Detection #${detection.id}</h6>
                            <small class="text-muted">
                                ${detection.lat ? detection.lat.toFixed(4) : 'N/A'}°N, 
                                ${detection.lon ? Math.abs(detection.lon).toFixed(4) : 'N/A'}°W
                            </small>
                        </div>
                        <div class="text-end">
                            <i class="fas ${statusIcon}"></i>
                            <br><small>${statusText}</small>
                        </div>
                    </div>
                    <div class="mt-2">
                        <small>
                            <strong>Intensity:</strong> ${detection.intensity_db.toFixed(1)} dB |
                            <strong>Area:</strong> ${detection.area_pixels}px
                            ${detection.nearest_distance_m ? `| <strong>Distance:</strong> ${Math.round(detection.nearest_distance_m)}m` : ''}
                        </small>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    listContainer.innerHTML = listHTML;
}

function focusOnDetection(detectionId) {
    const detection = detections.find(d => d.id === detectionId);
    if (detection && detection.lat && detection.lon) {
        map.setView([detection.lat, detection.lon], 10);
        
        // Find and open the popup for this detection
        detectionsLayer.eachLayer(layer => {
            if (layer.getLatLng && 
                Math.abs(layer.getLatLng().lat - detection.lat) < 0.0001 &&
                Math.abs(layer.getLatLng().lng - detection.lon) < 0.0001) {
                layer.openPopup();
            }
        });
    }
}

function showDetectionModal(detection) {
    const modal = new bootstrap.Modal(document.getElementById('detectionModal'));
    const modalBody = document.getElementById('modal-body');
    
    const aisInfo = detection.has_ais_match ? `
        <div class="alert alert-success">
            <h6><i class="fas fa-link"></i> AIS Match Found</h6>
            <p><strong>MMSI:</strong> ${detection.nearest_mmsi}</p>
            <p><strong>Distance:</strong> ${Math.round(detection.nearest_distance_m)}m</p>
            ${detection.nearest_time_delta_s ? `<p><strong>Time Delta:</strong> ${Math.round(detection.nearest_time_delta_s)}s</p>` : ''}
            ${detection.ais_timestamp ? `<p><strong>AIS Timestamp:</strong> ${new Date(detection.ais_timestamp).toLocaleString()}</p>` : ''}
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
            <h6>Detection #${detection.id}</h6>
            
            <div class="row mb-3">
                <div class="col-6">
                    <strong>Position:</strong><br>
                    ${detection.lat ? detection.lat.toFixed(6) : 'N/A'}°N<br>
                    ${detection.lon ? Math.abs(detection.lon).toFixed(6) : 'N/A'}°W
                </div>
                <div class="col-6">
                    <strong>Pixel Position:</strong><br>
                    Row: ${Math.round(detection.row)}<br>
                    Col: ${Math.round(detection.col)}
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
            
            ${aisInfo}
            
            <div class="mt-3">
                <button class="btn btn-primary" onclick="focusOnDetection(${detection.id}); bootstrap.Modal.getInstance(document.getElementById('detectionModal')).hide();">
                    <i class="fas fa-map-marker-alt"></i> Show on Map
                </button>
            </div>
        </div>
    `;
    
    modal.show();
}

function showNotification(message, type) {
    // Create a simple toast notification
    const toast = document.createElement('div');
    toast.className = `alert alert-${type === 'success' ? 'success' : 'danger'} alert-dismissible fade show position-fixed`;
    toast.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    toast.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(toast);
    
    // Auto-remove after 5 seconds
    setTimeout(() => {
        if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
        }
    }, 5000);
}

// Utility function to make detection data globally accessible for popup callbacks
window.showDetectionModal = showDetectionModal;
window.focusOnDetection = focusOnDetection;
