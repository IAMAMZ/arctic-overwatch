'use client';

import { useState, useEffect } from 'react';
import { Ship, Link, AlertTriangle, BarChart3, Upload, Cloud } from 'lucide-react';

interface Detection {
  id: number;
  lat: number;
  lon: number;
  intensity_db: number;
  area_pixels: number;
  has_ais_match: boolean;
  nearest_mmsi?: string;
  nearest_distance_m?: number;
}

interface Stats {
  totalDetections: number;
  aisMatches: number;
  darkVessels: number;
  avgIntensity: number;
}

export default function Home() {
  const [stats, setStats] = useState<Stats>({
    totalDetections: 0,
    aisMatches: 0,
    darkVessels: 0,
    avgIntensity: 0
  });
  const [detections, setDetections] = useState<Detection[]>([]);
  const [uploadParams, setUploadParams] = useState({
    sigma: 3.0,
    window: 15,
    minArea: 3
  });
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    loadDetections();
  }, []);

  const loadDetections = async () => {
    try {
      const response = await fetch('http://localhost:5000/api/detections');
      const data = await response.json();
      
      if (data.detections) {
        setDetections(data.detections);
        updateStats(data.detections);
      }
    } catch (error) {
      console.error('Error loading detections:', error);
    }
  };

  const updateStats = (detectionData: Detection[]) => {
    const totalDetections = detectionData.length;
    const aisMatches = detectionData.filter(d => d.has_ais_match).length;
    const darkVessels = totalDetections - aisMatches;
    const avgIntensity = detectionData.length > 0 ? 
      detectionData.reduce((sum, d) => sum + d.intensity_db, 0) / detectionData.length : 0;
    
    setStats({
      totalDetections,
      aisMatches,
      darkVessels,
      avgIntensity
    });
  };

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('sigma', uploadParams.sigma.toString());
    formData.append('window', uploadParams.window.toString());
    formData.append('min_area_pixels', uploadParams.minArea.toString());
    
    setUploading(true);
    
    try {
      const response = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData
      });
      
      const data = await response.json();
      
      if (data.success) {
        setDetections(data.detections);
        updateStats(data.detections);
        // Trigger map refresh by posting message to iframe
        const iframe = document.querySelector('iframe') as HTMLIFrameElement;
        if (iframe) {
          iframe.contentWindow?.location.reload();
        }
      }
    } catch (error) {
      console.error('Upload failed:', error);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
      handleFileUpload(files[0]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900">
      {/* Header */}
      <div className="bg-black/20 backdrop-blur-sm border-b border-white/10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-400 to-purple-500 rounded-lg flex items-center justify-center">
                <Ship className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-white">Arctic Overwatch</h1>
                <p className="text-sm text-gray-300">SAR Vessel Detection System</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-6">
          <div className="bg-gradient-to-r from-blue-500 to-blue-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-blue-100 text-sm font-medium">Total Detections</p>
                <p className="text-3xl font-bold">{stats.totalDetections}</p>
              </div>
              <Ship className="w-8 h-8 text-blue-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-green-100 text-sm font-medium">AIS Matches</p>
                <p className="text-3xl font-bold">{stats.aisMatches}</p>
              </div>
              <Link className="w-8 h-8 text-green-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-red-500 to-red-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-red-100 text-sm font-medium">Potential Dark Vessels</p>
                <p className="text-3xl font-bold">{stats.darkVessels}</p>
              </div>
              <AlertTriangle className="w-8 h-8 text-red-200" />
            </div>
          </div>
          
          <div className="bg-gradient-to-r from-purple-500 to-purple-600 rounded-xl p-6 text-white">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-purple-100 text-sm font-medium">Avg Intensity (dB)</p>
                <p className="text-3xl font-bold">{stats.avgIntensity.toFixed(1)}</p>
              </div>
              <BarChart3 className="w-8 h-8 text-purple-200" />
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
          {/* Map Section */}
          <div className="lg:col-span-3">
            <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20 overflow-hidden">
              <div className="bg-gradient-to-r from-blue-600 to-purple-600 px-6 py-4">
                <h2 className="text-lg font-semibold text-white flex items-center">
                  <Ship className="w-5 h-5 mr-2" />
                  Detection Map
                </h2>
              </div>
              <div className="h-[600px] relative">
                <iframe
                  src="http://localhost:5000"
                  className="w-full h-full border-none"
                  title="Arctic Overwatch Map"
                  style={{ 
                    filter: 'hue-rotate(0deg) contrast(1.1) brightness(1.05)',
                    background: 'transparent'
                  }}
                />
              </div>
            </div>
          </div>

          {/* Sidebar */}
          <div className="space-y-6">
            {/* Upload Section */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20">
              <div className="bg-gradient-to-r from-indigo-600 to-purple-600 px-6 py-4">
                <h3 className="text-lg font-semibold text-white flex items-center">
                  <Upload className="w-5 h-5 mr-2" />
                  Upload SAR Image
                </h3>
              </div>
              <div className="p-6">
                <div
                  className={`border-2 border-dashed rounded-lg p-8 text-center transition-all duration-200 cursor-pointer ${
                    dragOver 
                      ? 'border-blue-400 bg-blue-400/10' 
                      : 'border-gray-400 hover:border-blue-400 hover:bg-blue-400/5'
                  }`}
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => document.getElementById('file-input')?.click()}
                >
                  <input
                    type="file"
                    id="file-input"
                    className="hidden"
                    accept=".tif,.tiff,.geotiff"
                    onChange={handleFileSelect}
                  />
                  
                  {uploading ? (
                    <div className="text-blue-400">
                      <div className="animate-spin w-8 h-8 border-2 border-blue-400 border-t-transparent rounded-full mx-auto mb-4"></div>
                      <p className="text-sm">Processing SAR image...</p>
                    </div>
                  ) : (
                    <div className="text-gray-300">
                      <Cloud className="w-12 h-12 mx-auto mb-4 text-gray-400" />
                      <p className="text-sm mb-2">Click to select or drag and drop SAR image</p>
                      <p className="text-xs text-gray-500">Supported formats: TIF, TIFF</p>
                    </div>
                  )}
                </div>

                {/* Upload Parameters */}
                <div className="mt-6 space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Sigma
                      </label>
                      <input
                        type="number"
                        value={uploadParams.sigma}
                        onChange={(e) => setUploadParams({...uploadParams, sigma: parseFloat(e.target.value)})}
                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                        step="0.1"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-2">
                        Window
                      </label>
                      <input
                        type="number"
                        value={uploadParams.window}
                        onChange={(e) => setUploadParams({...uploadParams, window: parseInt(e.target.value)})}
                        className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                      />
                    </div>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Min Area
                    </label>
                    <input
                      type="number"
                      value={uploadParams.minArea}
                      onChange={(e) => setUploadParams({...uploadParams, minArea: parseInt(e.target.value)})}
                      className="w-full px-3 py-2 bg-white/10 border border-white/20 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                  </div>
                </div>
              </div>
            </div>

            {/* Detection Details */}
            <div className="bg-white/10 backdrop-blur-sm rounded-xl border border-white/20">
              <div className="bg-gradient-to-r from-emerald-600 to-teal-600 px-6 py-4">
                <h3 className="text-lg font-semibold text-white flex items-center">
                  <BarChart3 className="w-5 h-5 mr-2" />
                  Detection Details
                </h3>
              </div>
              <div className="p-6 max-h-96 overflow-y-auto">
                {detections.length === 0 ? (
                  <p className="text-gray-400 text-center py-8">No detections found</p>
                ) : (
                  <div className="space-y-3">
                    {detections.slice(0, 10).map((detection, index) => (
                      <div 
                        key={detection.id} 
                        className={`p-4 rounded-lg border-l-4 ${
                          detection.has_ais_match 
                            ? 'bg-green-900/20 border-green-400' 
                            : 'bg-red-900/20 border-red-400'
                        }`}
                      >
                        <div className="flex items-start justify-between">
                          <div>
                            <h4 className="text-sm font-medium text-white">
                              Detection #{detection.id}
                            </h4>
                            <p className="text-xs text-gray-400 mt-1">
                              {detection.lat?.toFixed(4)}°N, {Math.abs(detection.lon)?.toFixed(4)}°W
                            </p>
                          </div>
                          <div className={`w-3 h-3 rounded-full ${
                            detection.has_ais_match ? 'bg-green-400' : 'bg-red-400'
                          }`}></div>
                        </div>
                        <div className="mt-2 text-xs space-y-1">
                          <div className="flex justify-between text-gray-300">
                            <span>Intensity:</span>
                            <span>{detection.intensity_db?.toFixed(1)} dB</span>
                          </div>
                          <div className="flex justify-between text-gray-300">
                            <span>Area:</span>
                            <span>{detection.area_pixels}px</span>
                          </div>
                          {detection.nearest_distance_m && (
                            <div className="flex justify-between text-gray-300">
                              <span>Distance:</span>
                              <span>{Math.round(detection.nearest_distance_m)}m</span>
                            </div>
                          )}
                        </div>
                        <div className={`mt-2 px-2 py-1 rounded text-xs font-medium ${
                          detection.has_ais_match 
                            ? 'bg-green-400/20 text-green-300' 
                            : 'bg-red-400/20 text-red-300'
                        }`}>
                          {detection.has_ais_match ? 'AIS Match' : 'No AIS Match'}
                        </div>
                      </div>
                    ))}
                    {detections.length > 10 && (
                      <p className="text-center text-gray-400 text-sm mt-4">
                        Showing first 10 of {detections.length} detections
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}