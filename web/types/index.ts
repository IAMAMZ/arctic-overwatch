// web/types/index.ts
export type Detection = {
  id: number;
  lat: number;
  lon: number;
  intensity_db: number;
  area_pixels: number;
  has_ais_match: boolean;
};

export type AnalyzeTopRegion = {
  region: string;
  count: number;
  radiusKm: number;
  centroid: { lat: number; lon: number };
};

export type AnalyzeResponse = {
  requestedRegion: string | null;
  resolvedRegion: string | null;
  bestRegion: string;
  bestRegionCount: number;
  topRegions: AnalyzeTopRegion[];
  totalDetections: number;
  narrative: string;
};
