import os
import re
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def parse_sentinel1_timestamp_from_path(image_path: str) -> Optional[datetime]:
	"""Extract UTC acquisition timestamp from Sentinel-1 filename if present.

	Examples of supported patterns include:
	- S1A_EW_GRDM_1SDH_20250915T180548_431B_N_1.tif -> 2025-09-15T18:05:48Z
	"""
	name = os.path.basename(image_path)
	match = re.search(r"(\d{8}T\d{6})", name)
	if not match:
		return None
	dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
	return dt.replace(tzinfo=timezone.utc)


def haversine_distance_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
	"""Great-circle distance in meters between two WGS84 lon/lat points."""
	R = 6371000.0
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	dphi = phi2 - phi1
	dlambda = math.radians(lon2 - lon1)
	a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
	c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
	return R * c


def detections_to_feature_collection(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
	"""Convert detections to a GeoJSON FeatureCollection in WGS84.

	Detections without lon/lat are skipped.
	"""
	features: List[Dict[str, Any]] = []
	for det in detections:
		lon = det.get("lon")
		lat = det.get("lat")
		if lon is None or lat is None:
			continue
		props = {k: v for k, v in det.items() if k not in ("lon", "lat")}
		features.append(
			{
				"type": "Feature",
				"geometry": {"type": "Point", "coordinates": [lon, lat]},
				"properties": props,
			}
		)
	return {"type": "FeatureCollection", "features": features}

