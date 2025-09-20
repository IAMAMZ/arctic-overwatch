Arctic Dark Vessel Detection (SAR + AIS)

Setup (Windows PowerShell)

1. Create a virtual environment and install dependencies:

   - python -m venv .venv
   - .\.venv\Scripts\python -m pip install --upgrade pip
   - .\.venv\Scripts\python -m pip install -r requirements.txt

2. Run the detector on your Sentinel-1 image. Example:

   - .\.venv\Scripts\python -m sar_app.cli --image "S1A_EW_GRDM_1SDH_20250915T180548_431B_N_1.tif" --out "output"

3. Optionally include AIS CSV to classify matches vs potential dark vessels:

   - .\.venv\Scripts\python -m sar_app.cli --image "S1A_EW_GRDM_1SDH_20250915T180548_431B_N_1.tif" --ais "ais.csv" --out "output" --time-window-min 60 --distance-m 3000

Outputs

- output/detections.csv: Tabular detections with pixel, map, and WGS84 coords
- output/detections.geojson: Point features (WGS84) for mapping

Notes

- The timestamp is parsed from the Sentinel-1 filename (e.g., 20250915T180548).
- If the input TIFF includes a valid CRS, detections are exported in WGS84.
- If no CRS is found, matching to AIS will be skipped and only pixel coordinates are provided.

