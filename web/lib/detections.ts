// web/lib/detections.ts
import { promises as fs } from "node:fs";
import path from "node:path";
import type { Detection } from "@/types";

export async function loadDetections(): Promise<Detection[]> {
  // web/ is cwd in Next server; step out to ../myoutput/detections.csv
  const csvPath = path.join(process.cwd(), "..", "myoutput", "detections.csv");
  const raw = await fs.readFile(csvPath, "utf8");
  return parseDetectionsCSV(raw);
}

function parseDetectionsCSV(csv: string): Detection[] {
  const lines = csv
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l.length > 0);

  if (lines.length <= 1) return [];

  const header = lines[0].split(",");
  const idx = (name: string) => header.indexOf(name);

  const idI = idx("id");
  const latI = idx("lat");
  const lonI = idx("lon");
  const intensityI = idx("intensity_db");
  const areaI = idx("area_pixels");
  const hasAIS = idx("has_ais_match");

  const data: Detection[] = [];

  for (let i = 1; i < lines.length; i++) {
    const cols = splitCSVLine(lines[i]);
    if (!cols || cols.length !== header.length) continue;

    const d: Detection = {
      id: numberOr(cols[idI], i),
      lat: numberOr(cols[latI], NaN),
      lon: numberOr(cols[lonI], NaN),
      intensity_db: numberOr(cols[intensityI], 0),
      area_pixels: numberOr(cols[areaI], 0),
      has_ais_match: toBool(cols[hasAIS]),
    };

    if (Number.isFinite(d.lat) && Number.isFinite(d.lon)) {
      data.push(d);
    }
  }
  return data;
}

function numberOr(x: string, fallback: number) {
  const v = Number(x);
  return Number.isFinite(v) ? v : fallback;
}

function toBool(x: string) {
  const t = (x || "").trim().toLowerCase();
  return t === "true" || t === "1" || t === "yes";
}

// naïve CSV splitter that respects empty fields; adjust if you expect quoted commas
function splitCSVLine(line: string): string[] {
  return line.split(",");
}
