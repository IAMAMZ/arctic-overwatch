from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import haversine_distance_m


def _vector_haversine_m(lon: float, lat: float, lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
	phi1 = np.radians(lat)
	phi2 = np.radians(lats)
	dphi = phi2 - phi1
	dlambda = np.radians(lons - lon)
	a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
	return 2.0 * 6371000.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def match_detections_to_ais(
	detections: List[Dict[str, Any]],
	ais: pd.DataFrame,
	detection_time: Optional[datetime],
	time_window_minutes: int = 60,
	distance_m: float = 3000.0,
) -> List[Dict[str, Any]]:
	"""Attach AIS match info to each detection.

	If detection_time is provided, AIS is filtered to a +/- window around it.
	A detection is considered matched if the nearest AIS position is within distance_m.
	"""
	results: List[Dict[str, Any]] = []
	if ais is None or len(ais) == 0:
		for det in detections:
			out = {**det}
			out.update({
				"has_ais_match": False,
				"nearest_mmsi": "",
				"nearest_distance_m": None,
				"nearest_time_delta_s": None,
				"ais_timestamp": None,
			})
			results.append(out)
		return results

	if detection_time is not None and "timestamp" in ais.columns and not ais["timestamp"].isna().all():
		start = detection_time - timedelta(minutes=time_window_minutes)
		end = detection_time + timedelta(minutes=time_window_minutes)
		cand = ais[(ais["timestamp"] >= start) & (ais["timestamp"] <= end)].copy()
		if len(cand) == 0:
			cand = ais.copy()
	else:
		cand = ais.copy()

	cand_lons = cand["lon"].to_numpy(dtype=float)
	cand_lats = cand["lat"].to_numpy(dtype=float)

	for det in detections:
		lon = det.get("lon")
		lat = det.get("lat")
		out = {**det}
		if lon is None or lat is None or np.isnan(lon) or np.isnan(lat):
			out.update({
				"has_ais_match": False,
				"nearest_mmsi": "",
				"nearest_distance_m": None,
				"nearest_time_delta_s": None,
				"ais_timestamp": None,
			})
			results.append(out)
			continue

		dists = _vector_haversine_m(float(lon), float(lat), cand_lons, cand_lats)
		if len(dists) == 0:
			out.update({
				"has_ais_match": False,
				"nearest_mmsi": "",
				"nearest_distance_m": None,
				"nearest_time_delta_s": None,
				"ais_timestamp": None,
			})
			results.append(out)
			continue

		k = int(np.nanargmin(dists))
		min_dist = float(dists[k])
		nearest = cand.iloc[k]
		matched = min_dist <= float(distance_m)
		time_delta_s: Optional[float] = None
		if "timestamp" in cand.columns and pd.notna(nearest.get("timestamp")) and detection_time is not None:
			time_delta_s = abs((nearest["timestamp"] - detection_time).total_seconds())

		out.update({
			"has_ais_match": bool(matched),
			"nearest_mmsi": str(nearest.get("mmsi") or ""),
			"nearest_distance_m": min_dist,
			"nearest_time_delta_s": time_delta_s,
			"ais_timestamp": nearest.get("timestamp") if "timestamp" in cand.columns else None,
		})
		results.append(out)

	return results

