// web/lib/regions.ts
export type Region = {
  name: string;
  centroid: { lat: number; lon: number };
  radiusKm: number;
  aliases?: string[];
};

export const REGIONS: Region[] = [
  {
    name: "Fram Strait",
    centroid: { lat: 79.0, lon: -5.0 },
    radiusKm: 400,
    aliases: ["fram", "fram strait", "between svalbard and greenland"],
  },
  {
    name: "Northeast Greenland",
    centroid: { lat: 80.4, lon: -20.5 },
    radiusKm: 350,
    aliases: ["ne greenland", "northeast greenland", "ne greenland shelf"],
  },
  {
    name: "Jan Mayen Area",
    centroid: { lat: 71.0, lon: -8.0 },
    radiusKm: 500,
    aliases: ["jan mayen", "jan-mayen"],
  },
  {
    name: "Greenland Sea",
    centroid: { lat: 77.5, lon: -15.0 },
    radiusKm: 450,
    aliases: ["greenland sea"],
  },
  {
    name: "Norwegian Sea",
    centroid: { lat: 70.5, lon: 5.0 },
    radiusKm: 600,
    aliases: ["norwegian sea"],
  },
  {
    name: "Svalbard",
    centroid: { lat: 78.5, lon: 16.0 },
    radiusKm: 350,
    aliases: ["spitsbergen", "svalbard"],
  },
];

export function resolveRegion(input: string) {
  const q = input.trim().toLowerCase();
  for (const r of REGIONS) {
    if (r.name.toLowerCase() === q) return r;
    if (r.aliases?.some((a) => a.toLowerCase() === q)) return r;
  }
  // fuzzy contains match
  for (const r of REGIONS) {
    if (r.name.toLowerCase().includes(q)) return r;
    if (r.aliases?.some((a) => a.toLowerCase().includes(q))) return r;
  }
  return null;
}
