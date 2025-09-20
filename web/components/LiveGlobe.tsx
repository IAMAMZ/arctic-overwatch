"use client";

import { useEffect, useRef, useState } from "react";

type ShipPoint = {
  lat: number;
  lng: number;
  ts: number;
  mmsi?: string;
  conf?: number;
};

type GJFeature = {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: { timestamp: string | number; mmsi?: string; conf?: number };
};
type GJ = { type: "FeatureCollection"; features: GJFeature[] };

export default function LiveGlobe({
  geojsonUrl = "/data/year_ships_water.geojson",
  simHoursPerSec = 24,
  pointSize = 0.9
}: {
  geojsonUrl?: string;
  simHoursPerSec?: number;
  pointSize?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<any>(null);
  const timerRef = useRef<number | null>(null);

  const [loaded, setLoaded] = useState(false);
  const [selected, setSelected] = useState<ShipPoint | null>(null);

  useEffect(() => {
    let isAlive = true;

    (async () => {
      if (!containerRef.current) return;

      const mod = await import("globe.gl");
      const GlobeModule: any = (mod as any).default ?? mod;

      const globe =
        typeof GlobeModule === "function" && !GlobeModule.prototype?.constructor
          ? GlobeModule()(containerRef.current)
          : new (GlobeModule as any)(containerRef.current);

      globe.enablePointerInteraction(true);

      globe
        .globeTileEngineUrl((x: number, y: number, l: number) =>
          `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${l}/${y}/${x}`
        )
        // IMPORTANT: disable merge so hover/click work
        .pointsMerge(false)
        .pointAltitude(0.02)
        .pointRadius(pointSize)
        .pointResolution(16) // smoother hit area
        .pointColor((d: ShipPoint) => (d.conf && d.conf < 0.5 ? "#00ff88" : "#ff3355"))
        .pointLabel((d: ShipPoint) => {
          const t = new Date(d.ts).toISOString();
          return `<div><b>MMSI:</b> ${d.mmsi ?? "—"}<br/><b>Time (UTC):</b> ${t}<br/><b>Conf:</b> ${d.conf ?? "—"}</div>`;
        })
        .pointsTransitionDuration(300)
        .showPointerCursor(true) // built-in cursor helper
        .onPointHover((d: ShipPoint | null) => {
          if (!containerRef.current) return;
          containerRef.current.style.cursor = d ? "pointer" : "grab";
        })
        .onPointClick((d: ShipPoint, _evt: MouseEvent, coords: { lat: number; lng: number; altitude: number }) => {
          console.log("Clicked:", d, coords);
          setSelected(d);
          globe.pointOfView({ lat: d.lat, lng: d.lng, altitude: 0.9 }, 900);
        });

      globeRef.current = globe;

      const resize = () => {
        globe.width(window.innerWidth);
        globe.height(window.innerHeight);
        globe.pointOfView({ lat: 72, lng: -95, altitude: 1.8 }, 0);
      };
      resize();
      window.addEventListener("resize", resize);

      // fetch data
      const res = await fetch(`${geojsonUrl}?t=${Date.now()}`, { cache: "no-store" });
      const gj: GJ = await res.json();
      if (!gj?.features) throw new Error("GeoJSON missing features array");

      const all: ShipPoint[] = gj.features
        .filter(f => f.geometry?.type === "Point")
        .map(f => {
          const [lng, lat] = f.geometry.coordinates;
          const raw = f.properties.timestamp;
          const ts = typeof raw === "number" ? raw : Date.parse(raw);
          return { lat, lng, ts, mmsi: f.properties.mmsi, conf: f.properties.conf };
        })
        .filter(p => Number.isFinite(p.ts))
        .sort((a, b) => a.ts - b.ts);

      if (!all.length) {
        console.warn("No valid points in GeoJSON.");
        setLoaded(true);
        return;
      }

      const t0 = all[0].ts;
      const t1 = all[all.length - 1].ts;

      const displayed: ShipPoint[] = [];
      globe.pointsData(displayed);

      const simMsPerSec = simHoursPerSec * 3600 * 1000;
      let simStartRT = performance.now();
      let simStartTs = t0;
      let idx = 0;

      const restart = () => {
        displayed.length = 0;
        globe.pointsData(displayed);
        simStartRT = performance.now();
        simStartTs = t0;
        idx = 0;
        setSelected(null);
      };

      const tick = () => {
        if (!isAlive) return;

        const rtNow = performance.now();
        const simNow = simStartTs + (rtNow - simStartRT) * (simMsPerSec / 1000);

        while (idx < all.length && all[idx].ts <= simNow) {
          displayed.push(all[idx]);
          idx++;
        }

        globe.pointsData(displayed);

        if (simNow >= t1) restart();
        timerRef.current = requestAnimationFrame(tick);
      };

      timerRef.current = requestAnimationFrame(tick);
      setLoaded(true);

      return () => {
        isAlive = false;
        if (timerRef.current) cancelAnimationFrame(timerRef.current);
        window.removeEventListener("resize", resize);
        if (containerRef.current) containerRef.current.innerHTML = "";
      };
    })().catch(err => {
      console.error(err);
      alert(err instanceof Error ? err.message : String(err));
    });

    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [geojsonUrl, simHoursPerSec, pointSize]);

  const fmt = (n?: number, digits = 3) => (typeof n === "number" ? n.toFixed(digits) : "—");

  return (
    <>
      <div
        ref={containerRef}
        aria-busy={!loaded}
        style={{ position: "fixed", inset: 0, overflow: "hidden", pointerEvents: "auto" }}
      />
      {selected && (
        <div
          style={{
            position: "fixed",
            right: 16,
            top: 16,
            width: 320,
            background: "rgba(0,0,0,0.72)",
            color: "#fff",
            padding: "12px 14px",
            borderRadius: 12,
            backdropFilter: "blur(4px)",
            boxShadow: "0 8px 24px rgba(0,0,0,0.25)",
            zIndex: 50
          }}
        >
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
            <strong>Ship Detection</strong>
            <button
              onClick={() => setSelected(null)}
              style={{ background: "transparent", color: "#fff", border: "none", cursor: "pointer", fontSize: 18 }}
              aria-label="Close"
            >
              ×
            </button>
          </div>
          <div style={{ marginTop: 8, fontSize: 14, lineHeight: 1.5 }}>
            <div><b>MMSI:</b> {selected.mmsi ?? "—"}</div>
            <div><b>Time (UTC):</b> {new Date(selected.ts).toISOString()}</div>
            <div><b>Confidence:</b> {typeof selected.conf === "number" ? selected.conf.toFixed(2) : "—"}</div>
            <div><b>Lat/Lon:</b> {fmt(selected.lat)}, {fmt(selected.lng)}</div>
          </div>
        </div>
      )}
    </>
  );
}
