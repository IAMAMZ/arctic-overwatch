from typing import Dict, List, Optional

import pandas as pd


def _pick(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
	lower_to_orig: Dict[str, str] = {c.lower(): c for c in df.columns}
	for name in candidates:
		orig = lower_to_orig.get(name.lower())
		if orig is not None:
			return orig
	return None


def load_ais_csv(csv_path: str) -> pd.DataFrame:
	"""Load and normalize an AIS CSV file into standard columns.

	Expected output columns:
	- mmsi (str)
	- lon (float)
	- lat (float)
	- timestamp (datetime64[ns, UTC], optional if absent in input)
	- sog (float, knots, optional)
	- cog (float, degrees, optional)
	"""
	df = pd.read_csv(csv_path)
	mmsi_col = _pick(df, ["mmsi", "mmsi_id", "user_id", "userid"])
	lon_col = _pick(df, ["lon", "longitude", "long", "x"])
	lat_col = _pick(df, ["lat", "latitude", "y"])
	ts_col = _pick(df, [
		"timestamp",
		"time",
		"datetime",
		"basedatetime",
		"basetime",
		"date",
		"position_time",
	])
	sog_col = _pick(df, ["sog", "speed", "speed_over_ground", "sog_knots"])
	cog_col = _pick(df, ["cog", "course", "course_over_ground"]) 

	if lon_col is None or lat_col is None:
		raise ValueError("AIS CSV must include longitude and latitude columns")

	res = pd.DataFrame()
	if mmsi_col is not None:
		res["mmsi"] = df[mmsi_col].astype(str)
	else:
		res["mmsi"] = ""
	res["lon"] = pd.to_numeric(df[lon_col], errors="coerce")
	res["lat"] = pd.to_numeric(df[lat_col], errors="coerce")

	if ts_col is not None:
		res["timestamp"] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")
	else:
		res["timestamp"] = pd.NaT

	if sog_col is not None:
		res["sog"] = pd.to_numeric(df[sog_col], errors="coerce")
	else:
		res["sog"] = pd.NA
	if cog_col is not None:
		res["cog"] = pd.to_numeric(df[cog_col], errors="coerce")
	else:
		res["cog"] = pd.NA

	res = res.dropna(subset=["lon", "lat"])  # require positions
	if "timestamp" in res.columns:
		try:
			res = res.sort_values("timestamp")
		except Exception:
			pass
	res = res.reset_index(drop=True)
	return res

