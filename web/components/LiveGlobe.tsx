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
  geojsonUrl = "/data/year_ships.geojson", // <-- fetches from /public/data/
  simHoursPerSec = 24,
  pointSize = 0.7,
}: {
  geojsonUrl?: string;
  simHoursPerSec?: number;
  pointSize?: number;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const globeRef = useRef<any>(null);
  const timerRef = useRef<number | null>(null);
  const [loaded, setLoaded] = useState(false);

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

      globe
        .globeTileEngineUrl((x: number, y: number, l: number) =>
          `https://tile.openstreetmap.org/${l}/${x}/${y}.png`
        )
        .pointsMerge(true)
        .pointAltitude(0.01)
        .pointRadius(pointSize)
        .pointColor((d: ShipPoint) =>
          d.conf && d.conf < 0.5 ? "#00ff88" : "#ff3355"
        )
        .pointLabel((d: ShipPoint) => {
          const t = new Date(d.ts).toISOString();
          return `<div><b>MMSI:</b> ${d.mmsi ?? "—"}<br/><b>Time (UTC):</b> ${t}<br/><b>Conf:</b> ${d.conf ?? "—"}</div>`;
        })
        .pointsTransitionDuration(300);

      globeRef.current = globe;

      const resize = () => {
        globe.width(window.innerWidth);
        globe.height(window.innerHeight);
        globe.pointOfView({ lat: 72, lng: -95, altitude: 1.8 }, 0);
      };
      resize();
      window.addEventListener("resize", resize);

      // --------- Fetch GeoJSON ---------
      const url = `${geojsonUrl}?t=${Date.now()}`;
      const res = await fetch(url, { cache: "no-store" });
      const gj: GJ = await res.json();
      if (!gj?.features) throw new Error("GeoJSON missing features array");
      // ---------------------------------

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

  return (
    <div
      ref={containerRef}
      aria-busy={!loaded}
      style={{ position: "fixed", inset: 0, overflow: "hidden" }}
    />
  );
}
