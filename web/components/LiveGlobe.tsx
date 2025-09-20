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

      // Three for custom objects (glowing spheres + halo sprite)
      const THREE = await import("three");

      const globe =
        typeof GlobeModule === "function" && !GlobeModule.prototype?.constructor
          ? GlobeModule()(containerRef.current)
          : new (GlobeModule as any)(containerRef.current);

      // === Space background + auto-rotate (paused on click) ===
      globe
        .backgroundImageUrl("//unpkg.com/three-globe/example/img/night-sky.png")
        .showAtmosphere(true);

      const controls = globe.controls();
      if (controls) {
        controls.autoRotate = true;
        controls.autoRotateSpeed = 0.35;
        controlsRef.current = controls;
      }

      // Add a bit more ambient light so emissive pops nicely
      const ambient = new THREE.AmbientLight(0xffffff, 1.0);
      globe.scene().add(ambient);

      // === Zoom-aware dot radius (smaller when zooming in) ===
      const BASE_ALT = 1.8;    // initial camera altitude
      const BASE_RADIUS = 0.9; // base angular radius (tweak to taste)
      const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v));
      let currAlt = BASE_ALT;
      const scaleFromAlt = (alt: number) => clamp(alt / BASE_ALT, 0.25, 2);
      const radiusAccessor = () => BASE_RADIUS * scaleFromAlt(currAlt);

      globe
        // Optional surface: satellite tiles under the starfield
        .globeTileEngineUrl((x: number, y: number, l: number) =>
          `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${l}/${y}/${x}`
        )
        // Points layer (kept for hover/click + the “dot” look)
        .pointsMerge(false)          // needed for hover/click
        .pointAltitude(0.02)
        .pointRadius(radiusAccessor) // zoom-aware
        .pointResolution(20)
        .pointColor((d: ShipPoint) =>
          d.conf && d.conf < 0.5 ? "rgba(0,255,170,0.95)" : "rgba(255,60,100,0.95)"
        )
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
          setSelected(d);
          if (controlsRef.current) controlsRef.current.autoRotate = false; // stop spin on selection
          // (optional zoom-to)
          globe.pointOfView({ lat: d.lat, lng: d.lng, altitude: 0.9 }, 700);
        });

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

      // === Load GeoJSON ===
      const res = await fetch(`${geojsonUrl}?t=${Date.now()}`, { cache: "no-store" });
      if (!res.ok) throw new Error(`Failed to load ${geojsonUrl} (${res.status})`);
      const gj: GJ = await res.json();
      if (!gj?.features) throw new Error("GeoJSON missing features array");

      // Normalize + sort by time
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

      // === “Techy” visual layer 1: Sonar rings on appearances ===
      // Use same data array we progressively append to.
      globe
        .ringsData([]) // start empty, we’ll keep reassigning
        .ringColor((d: ShipPoint) =>
          d.conf && d.conf < 0.5 ? "rgba(0,255,200,0.65)" : "rgba(255,120,160,0.65)"
        )
        .ringMaxRadius(10)          // degrees
        .ringPropagationSpeed(2.3)   // degrees/sec
        .ringRepeatPeriod(0);        // single ping per point

      // === “Techy” visual layer 2: Glowing spheres (3D objects) at detections ===
      // Build a small glowing orb + a halo sprite, then clone per point.
      const buildGlowSprite = () => {
        const size = 128;
        const canvas = document.createElement("canvas");
        canvas.width = size;
        canvas.height = size;
        const ctx = canvas.getContext("2d")!;
        const g = ctx.createRadialGradient(size/2, size/2, 0, size/2, size/2, size/2);
        g.addColorStop(0.0, "rgba(0,255,220,0.9)");
        g.addColorStop(0.4, "rgba(0,255,220,0.25)");
        g.addColorStop(1.0, "rgba(0,255,220,0.0)");
        ctx.fillStyle = g;
        ctx.fillRect(0, 0, size, size);
        const tex = new THREE.Texture(canvas);
        tex.needsUpdate = true;
        const material = new THREE.SpriteMaterial({ map: tex, depthWrite: false, transparent: true });
        const sprite = new THREE.Sprite(material);
        sprite.scale.set(0.35, 0.35, 1); // adjust halo size
        return sprite;
      };

      const buildShipGlow = (d: ShipPoint) => {
        const orbGeom = new THREE.SphereGeometry(0.035, 20, 20);
        const color = d.conf && d.conf < 0.5 ? 0x00ffb0 : 0xff3c64;
        const mat = new THREE.MeshStandardMaterial({
          color,
          emissive: color,
          emissiveIntensity: 1.2,
          metalness: 0.2,
          roughness: 0.3
        });
        const orb = new THREE.Mesh(orbGeom, mat);
        const halo = buildGlowSprite();
        const group = new THREE.Group();
        group.add(orb);
        group.add(halo);
        return group;
      };

      globe
        .objectsData([]) // start empty; we'll mirror `displayed`
        .objectLat((d: ShipPoint) => d.lat)
        .objectLng((d: ShipPoint) => d.lng)
        .objectAltitude(0.028)
        .objectThreeObject((d: ShipPoint) => buildShipGlow(d));

      // === Playback over the year (progressively add items) ===
      const t0 = all[0].ts;
      const t1 = all[all.length - 1].ts;

      const displayed: ShipPoint[] = [];
      globe.pointsData(displayed);   // dots
      globe.ringsData(displayed);    // rings follow same array
      globe.objectsData(displayed);  // glowing spheres mirror the array

      const simMsPerSec = simHoursPerSec * 3600 * 1000;
      let simStartRT = performance.now();
      let simStartTs = t0;
      let idx = 0;

      const restart = () => {
        displayed.length = 0;
        globe.pointsData(displayed);
        globe.ringsData(displayed);
        globe.objectsData(displayed);
        simStartRT = performance.now();
        simStartTs = t0;
        idx = 0;
        // do not auto-resume rotate here—user intent stays respected
      };

      const tick = () => {
        if (!isAlive) return;

        const rtNow = performance.now();
        const simNow = simStartTs + (rtNow - simStartRT) * (simMsPerSec / 1000);

        // append any new detections whose ts <= simNow
        while (idx < all.length && all[idx].ts <= simNow) {
          displayed.push(all[idx]);
          idx++;
        }

        // rebind data so rings/objects/dots get the new ones
        globe.pointsData(displayed);
        globe.ringsData(displayed);
        globe.objectsData(displayed);

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

  const closePanel = () => {
    setSelected(null);
    if (controlsRef.current) controlsRef.current.autoRotate = true; // resume spin when closing
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
