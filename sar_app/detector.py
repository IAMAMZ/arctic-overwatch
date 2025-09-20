from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import rasterio
from rasterio.transform import xy as transform_xy
from pyproj import Transformer
from scipy.ndimage import (
	binary_closing,
	binary_opening,
	center_of_mass,
	find_objects,
	label,
	uniform_filter,
)


@dataclass
class Detection:
	id: int
	row: float
	col: float
	area_pixels: int
	intensity_db: float
	x: Optional[float]
	y: Optional[float]
	lon: Optional[float]
	lat: Optional[float]


def _read_primary_band(image_path: str) -> Tuple[np.ndarray, Any, Any, Optional[float]]:
	with rasterio.open(image_path) as ds:
		arr = ds.read(1).astype(np.float32)
		return arr, ds.transform, ds.crs, ds.nodata


def _to_decibel(arr: np.ndarray) -> np.ndarray:
	return 10.0 * np.log10(arr + 1.0e-6)


def detect_vessels(
	image_path: str,
	sigma: float = 3.0,
	window: int = 15,
	min_area_pixels: int = 3,
) -> List[Dict[str, Any]]:
	"""Detect bright targets in SAR imagery using an adaptive threshold.

	Returns a list of detection dicts with pixel, map, and WGS84 coordinates when possible.
	"""
	arr, transform, crs, _ = _read_primary_band(image_path)
	db = _to_decibel(arr)
	if not np.isfinite(db).all():
		db = np.nan_to_num(db, nan=-100.0, posinf=-100.0, neginf=-100.0)

	mean = uniform_filter(db, size=window, mode="reflect")
	mean_sq = uniform_filter(db ** 2, size=window, mode="reflect")
	var = np.maximum(mean_sq - mean ** 2, 1.0e-6)
	std = np.sqrt(var)
	threshold = mean + sigma * std

	mask = db > threshold
	mask = binary_opening(mask, structure=np.ones((3, 3), dtype=bool))
	mask = binary_closing(mask, structure=np.ones((3, 3), dtype=bool))

	labeled, num = label(mask)
	if num == 0:
		return []

	slices = find_objects(labeled)
	transformer: Optional[Transformer] = None
	lonlat_possible = crs is not None
	lonlat_native = False
	if lonlat_possible:
		crs_str = str(crs).lower()
		lonlat_native = "4326" in crs_str or "epsg:4326" in crs_str
		if not lonlat_native:
			try:
				transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
			except Exception:
				lonlat_possible = False

	detections: List[Dict[str, Any]] = []
	idx = 0
	for i, slc in enumerate(slices):
		if slc is None:
			continue
		label_id = i + 1
		region_mask = labeled[slc] == label_id
		area = int(region_mask.sum())
		if area < min_area_pixels:
			continue

		local_db = db[slc]
		com_r, com_c = center_of_mass(local_db, labels=region_mask.astype(np.uint8), index=1)
		row = float(slc[0].start + com_r)
		col = float(slc[1].start + com_c)
		peak_db = float(local_db[region_mask].max())

		x, y = transform_xy(transform, row, col)
		lon: Optional[float] = None
		lat: Optional[float] = None
		if lonlat_possible:
			if lonlat_native:
				lon, lat = float(x), float(y)
			elif transformer is not None:
				lon, lat = transformer.transform(x, y)

		detections.append(
			{
				"id": idx,
				"row": row,
				"col": col,
				"area_pixels": area,
				"intensity_db": peak_db,
				"x": float(x),
				"y": float(y),
				"lon": float(lon) if lon is not None else None,
				"lat": float(lat) if lat is not None else None,
			}
		)
		idx += 1

	return detections

