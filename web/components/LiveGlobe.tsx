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
  simHoursPerSec = 24
}: {
  geojsonUrl?: string;
  simHoursPerSec?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<any>(null);
  const controlsRef = useRef<any>(null);
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

      // --- Space background + gentle auto-rotate (will pause on selection) ---
      globe
        .backgroundImageUrl("//unpkg.com/three-globe/example/img/night-sky.png")
        .showAtmosphere(true);

      const controls = globe.controls();
      if (controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.35;
        controlsRef.current = controls;
      }

      // --- Zoom-aware dot radius (smaller when zooming in) ---
      const BASE_ALT = 1.8;    // initial camera altitude
      const BASE_RADIUS = 0.9; // base angular radius
      const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
      let currAlt = BASE_ALT;
      const scaleFromAlt = (alt: number) => clamp(alt / BASE_ALT, 0.25, 2);
      const radiusAccessor = () => BASE_RADIUS * scaleFromAlt(currAlt);

      globe
        // Optional surface detail under the starfield
        .globeTileEngineUrl((x: number, y: number, l: number) =>
          `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${l}/${y}/${x}`
        )
        // Points layer (merge=false so hover/click work)
        .pointsMerge(false)
        .pointAltitude(0.02)
        .pointRadius(radiusAccessor)
        .pointResolution(16)
        .pointColor((d: ShipPoint) => (d.conf && d.conf < 0.5 ? "#00ff88" : "#ff3355"))
        .pointLabel((d: ShipPoint) => {
          const t = new Date(d.ts).toISOString();
          return `<div><b>MMSI:</b> ${d.mmsi ?? "—"}<br/><b>Time (UTC):</b> ${t}<br/><b>Conf:</b> ${d.conf ?? "—"}</div>`;
        })
        .pointsTransitionDuration(300)
        .showPointerCursor(true)
        .onPointHover((d: ShipPoint | null) => {
          if (!containerRef.current) return;
          containerRef.current.style.cursor = d ? "pointer" : "grab";
        })
        .onPointClick((d: ShipPoint) => {
          setSelected(d);                 // show info
          if (controlsRef.current) {
            controlsRef.current.autoRotate = false; // ⛔ stop spin on click
          }
        });

      // keep radius in sync with camera zoom
      globe.onZoom(({ altitude }: { lat: number; lng: number; altitude: number }) => {
        currAlt = altitude;
        globe.pointRadius(radiusAccessor).pointsTransitionDuration(0);
      });

      globeRef.current = globe;

      // Fit viewport + initial POV
      const resize = () => {
        globe.width(window.innerWidth);
        globe.height(window.innerHeight);
        globe.pointOfView({ lat: 72, lng: -95, altitude: BASE_ALT }, 0);
      };
      resize();
      window.addEventListener("resize", resize);

      // ---- Load GeoJSON ----
      const res = await fetch(`${geojsonUrl}?t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Failed to load ${geojsonUrl} (${res.status})`);
      const gj: GJ = await res.json();
      if (!gj?.features) throw new Error("GeoJSON missing features array");

      // Normalize + sort
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

      // ---- Playback of dots over the year ----
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
        // don't auto-resume spin here; user intent is focused on selection flow
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

      // Cleanup
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
  }, [geojsonUrl, simHoursPerSec]);

  const fmt = (n?: number, digits = 3) => (typeof n === "number" ? n.toFixed(digits) : "—");

  // When the info panel closes, resume auto-rotate
  const closePanel = () => {
    setSelected(null);
    if (controlsRef.current) {
      controlsRef.current.autoRotate = true; // ▶️ resume spin when panel closes
    }
  };

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
              onClick={closePanel}
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
