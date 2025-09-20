"use client";

import { useEffect, useState } from "react";
import { BarChart3, Ship, Eye, Zap } from "lucide-react";

type Stats = {
  total_detections: number;
  ais_matches: number;
  dark_vessels: number;
  avg_intensity: number;
};

export default function Home() {
  const [stats, setStats] = useState<Stats>({
    total_detections: 0,
    ais_matches: 0,
    dark_vessels: 0,
    avg_intensity: 0,
  });

  useEffect(() => {
    async function loadStats() {
      try {
        const res = await fetch("http://localhost:5000/api/stats");
        const data = await res.json();
        setStats(data);
      } catch (err) {
        console.error("Failed to load stats:", err);
      }
    }
    loadStats();
  }, []);

  return (
    <div className="min-h-screen bg-gray-900 text-white flex flex-col">
      {/* Header */}
      <header className="bg-gray-800/80 backdrop-blur border-b border-gray-700 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 flex items-center justify-center rounded-lg bg-gradient-to-r from-blue-500 to-cyan-500 shadow-lg">
            <Ship className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold bg-gradient-to-r from-blue-400 to-cyan-400 bg-clip-text text-transparent">
              Arctic Overwatch
            </h1>
            <p className="text-sm text-gray-400">SAR Vessel Detection System</p>
          </div>
        </div>
      </header>

      {/* Stats section */}
      <section className="grid grid-cols-4 gap-6 p-6 bg-gray-800/40 border-b border-gray-700">
        <StatCard
          label="Total Detections"
          value={stats.total_detections.toLocaleString()}
          icon={<BarChart3 className="w-6 h-6 text-blue-400" />}
          color="from-blue-600/20 to-blue-700/20 border-blue-500/30 text-blue-400"
        />
        <StatCard
          label="AIS Matches"
          value={stats.ais_matches.toLocaleString()}
          icon={<Eye className="w-6 h-6 text-green-400" />}
          color="from-green-600/20 to-green-700/20 border-green-500/30 text-green-400"
        />
        <StatCard
          label="Dark Vessels"
          value={stats.dark_vessels.toLocaleString()}
          icon={<Zap className="w-6 h-6 text-red-400" />}
          color="from-red-600/20 to-red-700/20 border-red-500/30 text-red-400"
        />
        <StatCard
          label="Avg Intensity (dB)"
          value={stats.avg_intensity.toFixed(1)}
          icon={<BarChart3 className="w-6 h-6 text-yellow-400" />}
          color="from-yellow-600/20 to-yellow-700/20 border-yellow-500/30 text-yellow-400"
        />
      </section>

      {/* Map from backend */}
      <main className="flex-1 relative">
        <iframe
          src="http://localhost:5000/"
          className="absolute inset-0 w-full h-full border-none"
          title="Detection Map"
        />
      </main>
    </div>
  );
}

/* Small reusable stat card */
function StatCard({
  label,
  value,
  icon,
  color,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <div
      className={`rounded-xl p-5 border bg-gradient-to-br ${color} flex flex-col space-y-3`}
    >
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-400 uppercase tracking-wider">
          {label}
        </span>
        {icon}
      </div>
      <div className="text-3xl font-bold">{value}</div>
    </div>
  );
}
