import argparse
import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd

from .ais import load_ais_csv
from .detector import detect_vessels
from .matcher import match_detections_to_ais
from .utils import detections_to_feature_collection, parse_sentinel1_timestamp_from_path


def _ensure_dir(path: str) -> None:
	if path and not os.path.exists(path):
		os.makedirs(path, exist_ok=True)


def run(
	image_path: str,
	out_dir: str,
	ais_csv: Optional[str] = None,
	sigma: float = 3.0,
	window: int = 15,
	min_area_pixels: int = 3,
	time_window_min: int = 60,
	distance_m: float = 3000.0,
) -> Dict[str, Any]:
	_ensure_dir(out_dir)
	print("Detecting vessels...")
	detections = detect_vessels(image_path, sigma=sigma, window=window, min_area_pixels=min_area_pixels)

	acq_time = parse_sentinel1_timestamp_from_path(image_path)
	print(f"Parsed acquisition time: {acq_time}")

	ais_df = None
	if ais_csv:
		print(f"Loading AIS from: {ais_csv}")
		try:
			ais_df = load_ais_csv(ais_csv)
			print(f"Loaded {len(ais_df)} AIS records")
		except Exception as e:
			print(f"Failed to load AIS CSV: {e}")

	if ais_df is not None:
		print("Matching detections to AIS...")
		results = match_detections_to_ais(
			detections,
			ais_df,
			detection_time=acq_time,
			time_window_minutes=time_window_min,
			distance_m=distance_m,
		)
	else:
		results = []
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

	# Save CSV
	df = pd.DataFrame(results)
	csv_path = os.path.join(out_dir, "detections.csv")
	df.to_csv(csv_path, index=False)
	print(f"Saved: {csv_path}")

	# Save GeoJSON
	fc = detections_to_feature_collection(results)
	geojson_path = os.path.join(out_dir, "detections.geojson")
	with open(geojson_path, "w", encoding="utf-8") as f:
		json.dump(fc, f)
	print(f"Saved: {geojson_path}")

	return {"csv": csv_path, "geojson": geojson_path, "count": len(results)}


def main() -> None:
	parser = argparse.ArgumentParser(description="SAR dark vessel detector (Sentinel-1 + optional AIS)")
	parser.add_argument("--image", required=True, help="Path to Sentinel-1 GeoTIFF")
	parser.add_argument("--out", required=True, help="Output directory")
	parser.add_argument("--ais", help="Optional AIS CSV path")
	parser.add_argument("--sigma", type=float, default=3.0, help="Sigma for adaptive threshold")
	parser.add_argument("--window", type=int, default=15, help="Window size (pixels) for local stats")
	parser.add_argument("--min-area", type=int, default=3, help="Minimum connected area in pixels")
	parser.add_argument("--time-window-min", type=int, default=60, help="Time window (+/- minutes) for AIS matching")
	parser.add_argument("--distance-m", type=float, default=3000.0, help="Max AIS match distance in meters")

	args = parser.parse_args()
	run(
		image_path=args.image,
		out_dir=args.out,
		ais_csv=args.ais,
		sigma=args.sigma,
		window=args.window,
		min_area_pixels=args.min_area,
		time_window_min=args.time_window_min,
		distance_m=args.distance_m,
	)


if __name__ == "__main__":
	main()

