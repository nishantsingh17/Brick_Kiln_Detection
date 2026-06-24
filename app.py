import os
import glob
import shutil
import zipfile
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import streamlit as st
from rasterio.transform import xy
from rasterio.features import shapes
from shapely.geometry import Polygon, shape

from gee_utils import download_sentinel_tiles, test_gee_connection
from geo_utils import (
    read_tiff_as_rgb,
    stretch_for_display,
    create_kaggle_style_tif_patches,
)
from predict_utils import BrickKilnDetector

# For interactive map
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import Draw
import json
import requests

# =========================
# App setup
# =========================

st.set_page_config(page_title="Brick Kiln Detection", layout="wide")

st.title("🧱 Brick Kiln Detection using GEE + YOLOv8-OBB + SAM")
st.markdown("---")

# =========================
# Constants
# =========================

OUTPUT_DIR = Path("outputs")
TILE_DIR = OUTPUT_DIR / "gee_tiles"
PATCH_ROOT_DIR = OUTPUT_DIR / "sentinel_like_tif_patches"
DETECTED_PATCH_DIR = OUTPUT_DIR / "detected_patches"

YOLO_MODEL = "models/best.pt"
SAM_MODEL = "models/sam_b.pt"

PATCH_SIZE = 128
OVERLAP = 30
YOLO_IMGSZ = 416
UTM_CRS = "EPSG:32644"

# =========================
# Helper Functions
# =========================

def reset_output_dirs():
    OUTPUT_DIR.mkdir(exist_ok=True)
    for p in [TILE_DIR, PATCH_ROOT_DIR, DETECTED_PATCH_DIR]:
        if p.exists():
            shutil.rmtree(p)
        p.mkdir(parents=True, exist_ok=True)

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))

def remove_duplicates(df, obb_geometries, sam_geometries, distance_m=40):
    if df.empty:
        return df, [], []

    df = df.copy()
    df["geom_index"] = range(len(df))
    df_sorted = df.sort_values("Confidence", ascending=False).reset_index(drop=True)

    kept_rows = []
    kept_obb = []
    kept_sam = []

    for _, row in df_sorted.iterrows():
        duplicate = False
        for kept in kept_rows:
            dist = haversine(
                row["Latitude"], row["Longitude"],
                kept["Latitude"], kept["Longitude"]
            )
            if dist < distance_m:
                duplicate = True
                break

        if not duplicate:
            geom_idx = int(row["geom_index"])
            kept_rows.append(row)
            kept_obb.append(obb_geometries[geom_idx])
            kept_sam.append(sam_geometries[geom_idx] if geom_idx < len(sam_geometries) else None)

    final_df = pd.DataFrame(kept_rows).reset_index(drop=True)
    final_df = final_df.drop(columns=["geom_index"])
    final_df["Kiln_ID"] = [f"BK_{i + 1:04d}" for i in range(len(final_df))]

    return final_df, kept_obb, kept_sam

def write_shapefile_zip(gdf, base_path):
    base_path = Path(base_path)
    shp_path = base_path.with_suffix(".shp")
    zip_path = base_path.with_suffix(".zip")

    gdf.to_file(shp_path)

    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in glob.glob(str(base_path) + ".*"):
            if file.endswith(".zip"):
                continue
            zipf.write(file, arcname=os.path.basename(file))

    return shp_path, zip_path

def sam_mask_to_polygon(mask_np, transform, crs):
    polygons = []
    for geom, value in shapes(mask_np.astype(np.uint8), mask=mask_np > 0, transform=transform):
        if value == 1:
            polygons.append(shape(geom))

    if not polygons:
        return None, np.nan

    gdf = gpd.GeoDataFrame(geometry=polygons, crs=crs)
    gdf_m = gdf.to_crs(UTM_CRS)
    largest_idx = gdf_m.area.idxmax()
    polygon = gdf.loc[largest_idx].geometry
    area_m2 = float(gdf_m.loc[largest_idx].geometry.area)
    return polygon, area_m2

def geocode_location(location_name):
    try:
        url = f"https://nominatim.openstreetmap.org/search?q={location_name}, India&format=json&limit=1"
        headers = {'User-Agent': 'BrickKilnDetectionApp/1.0'}
        response = requests.get(url, headers=headers)
        data = response.json()
        
        if data:
            lat = float(data[0]['lat'])
            lon = float(data[0]['lon'])
            return lat, lon
        else:
            return None, None
    except Exception as e:
        st.error(f"Geocoding error: {e}")
        return None, None

def find_district_column(gdf):
    possible_names = ['district', 'DISTRICT', 'NAME', 'name', 'District', 'Name', 
                     'DIST', 'dist', 'dtname', 'DTNAME', 'DISTNAME', 'DistName',
                     'DISTRICT_NAME', 'district_name', 'DISTRICTNAME', 'DISTRICT_NA']
    
    for col in possible_names:
        if col in gdf.columns:
            return col
    
    for col in gdf.columns:
        if col.upper() in ['DISTRICT', 'NAME', 'DIST', 'DTNAME', 'DISTNAME', 'DISTRICT_NAME']:
            return col
    
    for col in gdf.columns:
        if gdf[col].dtype == 'object':
            return col
    
    return gdf.columns[0]

@st.cache_data
def load_district_data():
    try:
        data_folders = [Path("Data"), Path("data")]
        
        gdf = None
        
        for data_folder in data_folders:
            if data_folder.exists():
                shp_files = list(data_folder.glob("*.shp"))
                
                if shp_files:
                    shp_path = shp_files[0]
                    
                    try:
                        gdf = gpd.read_file(shp_path)
                        st.success(f"✅ Loaded {len(gdf)} districts from {shp_path.name}")
                        break
                    except Exception as e:
                        st.error(f"Error loading {shp_path.name}: {e}")
                        continue
        
        if gdf is None:
            st.warning("No shapefile (.shp) found in 'Data' or 'data' folder")
            return None
        
        if str(gdf.crs) != "EPSG:4326":
            st.info("Converting shapefile to WGS84 (latitude/longitude)...")
            gdf = gdf.to_crs("EPSG:4326")
            st.success("✅ Shapefile converted to WGS84")
        
        district_col = find_district_column(gdf)
        st.session_state.district_col = district_col
        gdf = gdf.rename(columns={district_col: 'district'})
        
        st.success(f"✅ Using district column: '{district_col}'")
        
        sample_districts = gdf['district'].head(3).tolist()
        st.info(f"📋 Sample districts: {', '.join(sample_districts)}")
        
        return gdf
        
    except Exception as e:
        st.error(f"❌ Failed to load district data: {str(e)}")
        return None

def get_district_geometry(gdf, district_name):
    district = gdf[gdf['district'] == district_name]
    if len(district) > 0:
        return district.geometry.iloc[0]
    return None

@st.cache_resource
def load_detector():
    return BrickKilnDetector(YOLO_MODEL, SAM_MODEL)

# =========================
# Initialize Session State
# =========================

if "aoi_bounds" not in st.session_state:
    st.session_state.aoi_bounds = [78.74, 26.42, 79.35, 27.01]
if "map_center" not in st.session_state:
    st.session_state.map_center = [26.7157, 79.0455]
if "selected_district" not in st.session_state:
    st.session_state.selected_district = None
if "district_geometry" not in st.session_state:
    st.session_state.district_geometry = None
if "selection_mode" not in st.session_state:
    st.session_state.selection_mode = "District"

# =========================
# Load District Data
# =========================

district_gdf = load_district_data()

# =========================
# Sidebar - Selection Mode
# =========================

st.sidebar.header("📍 AOI Selection Mode")

selection_mode = st.sidebar.radio(
    "Choose how to select area:",
    ["🏛️ Select District", "✏️ Draw Rectangle on Map", "🔍 Search Location"],
    index=0 if st.session_state.selection_mode == "District" else 1
)

st.session_state.selection_mode = selection_mode

# =========================
# Option 1: District Selection
# =========================
if selection_mode == "🏛️ Select District":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select District")
    
    if district_gdf is not None:
        district_list = sorted(district_gdf['district'].unique())
        
        selected_district = st.sidebar.selectbox(
            "Choose a District",
            ["-- Select District --"] + district_list,
            index=0,
            key="district_select"
        )
        
        if selected_district != "-- Select District --":
            st.session_state.selected_district = selected_district
            st.session_state.district_geometry = get_district_geometry(district_gdf, selected_district)
            
            if st.session_state.district_geometry is not None:
                bounds = st.session_state.district_geometry.bounds
                st.session_state.aoi_bounds = [bounds[0], bounds[1], bounds[2], bounds[3]]
                st.session_state.map_center = [(bounds[1] + bounds[3]) / 2, (bounds[0] + bounds[2]) / 2]
                
                width_km = (bounds[2] - bounds[0]) * 111
                height_km = (bounds[3] - bounds[1]) * 111
                
                st.sidebar.markdown("---")
                st.sidebar.subheader("📊 District Info")
                st.sidebar.write(f"**Name:** {selected_district}")
                st.sidebar.write(f"**Width:** {width_km:.1f} km")
                st.sidebar.write(f"**Height:** {height_km:.1f} km")
                st.sidebar.write(f"**Area:** {width_km * height_km:.1f} km²")
                
                # Show area classification
                area_km2 = width_km * height_km
                if area_km2 < 500:
                    st.sidebar.success("📱 Small area - Direct download")
                else:
                    st.sidebar.warning("📦 Large area - Will use Google Drive export")
        else:
            st.session_state.selected_district = None
            st.session_state.district_geometry = None
    else:
        st.sidebar.error("❌ No shapefile loaded")

# =========================
# Option 2: Draw Rectangle on Map
# =========================
elif selection_mode == "✏️ Draw Rectangle on Map":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Draw Rectangle")
    st.sidebar.info("🖱️ Click the rectangle icon (□) on the map below, then draw your AOI. Double-click to finish.")
    
    if st.sidebar.button("🔄 Reset to Default View", use_container_width=True):
        st.session_state.aoi_bounds = [78.74, 26.42, 79.35, 27.01]
        st.session_state.map_center = [26.7157, 79.0455]
        st.session_state.selected_district = None
        st.session_state.district_geometry = None
        st.rerun()
    
    st.session_state.selected_district = None
    st.session_state.district_geometry = None

# =========================
# Option 3: Search Location
# =========================
elif selection_mode == "🔍 Search Location":
    st.sidebar.markdown("---")
    st.sidebar.subheader("Search Location")
    
    place_name = st.sidebar.text_input("Enter city/town name", placeholder="e.g., Kanpur, Lucknow, Agra")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        if st.button("🔍 Search", use_container_width=True):
            if place_name:
                with st.spinner(f"Searching..."):
                    lat, lon = geocode_location(place_name)
                    if lat and lon:
                        st.session_state.map_center = [lat, lon]
                        st.success(f"Found: {place_name}")
                        st.rerun()
                    else:
                        st.error("Location not found")
    
    with col2:
        if st.button("📦 Set 10km Box", use_container_width=True):
            if place_name:
                with st.spinner(f"Searching..."):
                    lat, lon = geocode_location(place_name)
                    if lat and lon:
                        delta = 0.05
                        st.session_state.aoi_bounds = [lon - delta, lat - delta, lon + delta, lat + delta]
                        st.session_state.map_center = [lat, lon]
                        st.session_state.selected_district = None
                        st.session_state.district_geometry = None
                        st.success(f"✅ Set 10km box around {place_name}")
                        st.rerun()
                    else:
                        st.error("Location not found")

# =========================
# Display Current AOI Info
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📍 Current AOI")

xmin, ymin, xmax, ymax = st.session_state.aoi_bounds

st.sidebar.write(f"**Min Longitude:** {xmin:.6f}")
st.sidebar.write(f"**Max Longitude:** {xmax:.6f}")
st.sidebar.write(f"**Min Latitude:** {ymin:.6f}")
st.sidebar.write(f"**Max Latitude:** {ymax:.6f}")

lon_range_km = (xmax - xmin) * 111
lat_range_km = (ymax - ymin) * 111
area_km2 = lon_range_km * lat_range_km
st.sidebar.write(f"**Area:** {area_km2:.2f} km²")

if st.session_state.selected_district:
    st.sidebar.info(f"📍 Using district: {st.session_state.selected_district}")

# =========================
# Date Selection
# =========================
st.sidebar.markdown("---")
st.sidebar.subheader("📅 Date Range")

start_date = st.sidebar.text_input("Start Date", "2023-01-01", help="Format: YYYY-MM-DD")
end_date = st.sidebar.text_input("End Date", "2025-06-01", help="Format: YYYY-MM-DD")

# =========================
# Prediction Settings
# =========================
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Detection Settings")

conf = st.sidebar.slider("YOLO Confidence Threshold", 0.05, 0.80, 0.55, 0.05)
duplicate_distance_m = st.sidebar.slider("Duplicate Removal Distance (m)", 10, 100, 40, 5)
use_sam = st.sidebar.checkbox("Generate SAM Mask Shapefile", value=True)

st.sidebar.markdown("### 🌍 Download Settings")
tile_size_deg = st.sidebar.selectbox(
    "GEE Tile Size (degrees) for small areas",
    [0.05, 0.08, 0.10, 0.15],
    index=1,
    help="Only used for areas < 500 km²"
)
cloud_pct = st.sidebar.slider("Cloud Percentage Filter", 10, 80, 50, 5)

# =========================
# Test GEE Connection Button
# =========================
st.sidebar.markdown("---")
if st.sidebar.button("🔧 Test GEE Connection", use_container_width=True):
    with st.spinner("Testing GEE connection..."):
        success = test_gee_connection()
        if success:
            st.sidebar.success("✅ GEE is working!")
        else:
            st.sidebar.error("❌ GEE connection failed")

# =========================
# Interactive Map
# =========================

st.subheader("🗺️ Interactive Map")

# Create map
m = folium.Map(
    location=st.session_state.map_center,
    zoom_start=11,
    tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    attr="Google Satellite"
)

# Add drawing controls ONLY if in rectangle mode
if selection_mode == "✏️ Draw Rectangle on Map":
    draw = Draw(
        draw_options={
            "polyline": False,
            "polygon": False,
            "circle": False,
            "marker": False,
            "circlemarker": False,
            "rectangle": True,
        },
        edit_options={"edit": False},
    )
    draw.add_to(m)
    st.info("✏️ **Drawing Mode Active** - Click the □ icon on the map, then click and drag to draw a rectangle. Double-click to finish.")

# Add current AOI rectangle
folium.Rectangle(
    bounds=[[ymin, xmin], [ymax, xmax]],
    color="red" if selection_mode == "✏️ Draw Rectangle on Map" else "green",
    weight=3,
    fill=True,
    fill_opacity=0.1,
    popup="Current AOI",
    tooltip="Detection Area"
).add_to(m)

# Add district boundary if district is selected
if st.session_state.selected_district and st.session_state.district_geometry is not None:
    folium.GeoJson(
        st.session_state.district_geometry,
        style_function=lambda x: {
            'color': '#FF4444', 
            'weight': 4, 
            'fillColor': '#FF4444',
            'fillOpacity': 0.15
        },
        highlight_function=lambda x: {'weight': 6, 'color': '#FF0000', 'fillOpacity': 0.3},
        popup=f"<b>{st.session_state.selected_district}</b><br>District Boundary",
        tooltip=f"{st.session_state.selected_district} District"
    ).add_to(m)

# Display the map
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    output = st_folium(m, width=700, height=500, key="main_map")

# Process rectangle drawing
if selection_mode == "✏️ Draw Rectangle on Map" and output and output.get("last_active_drawing"):
    draw_data = output["last_active_drawing"]
    if draw_data and "geometry" in draw_data:
        coords = draw_data["geometry"]["coordinates"][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        xmin_new, xmax_new = min(lons), max(lons)
        ymin_new, ymax_new = min(lats), max(lats)
        st.session_state.aoi_bounds = [xmin_new, ymin_new, xmax_new, ymax_new]
        st.session_state.map_center = [(ymin_new + ymax_new) / 2, (xmin_new + xmax_new) / 2]
        st.session_state.selected_district = None
        st.session_state.district_geometry = None
        st.success(f"✅ Rectangle drawn! Bounds: Lon [{xmin_new:.4f} to {xmax_new:.4f}], Lat [{ymin_new:.4f} to {ymax_new:.4f}]")
        st.rerun()

# =========================
# Manual Lat/Long Input
# =========================
st.subheader("📝 Manual AOI Input")

col1, col2, col3, col4 = st.columns(4)

with col1:
    manual_xmin = st.number_input("Min Longitude", value=xmin, format="%.6f", step=0.01)

with col2:
    manual_xmax = st.number_input("Max Longitude", value=xmax, format="%.6f", step=0.01)

with col3:
    manual_ymin = st.number_input("Min Latitude", value=ymin, format="%.6f", step=0.01)

with col4:
    manual_ymax = st.number_input("Max Latitude", value=ymax, format="%.6f", step=0.01)

if st.button("Update AOI from Values", use_container_width=True):
    st.session_state.aoi_bounds = [manual_xmin, manual_ymin, manual_xmax, manual_ymax]
    st.session_state.map_center = [(manual_ymin + manual_ymax) / 2, (manual_xmin + manual_xmax) / 2]
    st.session_state.selected_district = None
    st.session_state.district_geometry = None
    st.success("✅ AOI updated!")
    st.rerun()

# =========================
# Detection Button
# =========================
st.markdown("---")

# Get final AOI bounds
xmin, ymin, xmax, ymax = st.session_state.aoi_bounds
district_geometry = st.session_state.district_geometry

if st.button("🚀 Detect Brick Kilns", type="primary", use_container_width=True):
    reset_output_dirs()

    with st.spinner("Loading YOLO and SAM models..."):
        detector = load_detector()
    st.success("✅ Models loaded successfully.")

    # Check if this is a large area that needs batch export
    if district_geometry is not None:
        bounds = district_geometry.bounds
        width_km = (bounds[2] - bounds[0]) * 111
        height_km = (bounds[3] - bounds[1]) * 111
        area_km2 = width_km * height_km
        
        if area_km2 > 500:
            st.warning(f"📦 **Large Area Detected: {area_km2:.1f} km²**")
            st.info("For large areas, GEE requires batch export to Google Drive instead of direct download.")
            st.markdown("---")

    st.info(f"📍 Detecting brick kilns...")
    if district_geometry is not None:
        st.info(f"District: {st.session_state.selected_district}")
    st.info(f"AOI bounds: Lon [{xmin:.4f} to {xmax:.4f}], Lat [{ymin:.4f} to {ymax:.4f}]")
    st.info(f"Area: {(xmax-xmin)*111 * (ymax-ymin)*111:.2f} km²")
    st.info(f"📅 Date range: {start_date} to {end_date}")
    st.info(f"☁️ Cloud filter: {cloud_pct}%")

    csv_path = OUTPUT_DIR / "kiln_results.csv"
    raw_csv_path = OUTPUT_DIR / "kiln_results_raw.csv"
    patch_csv_path = OUTPUT_DIR / "patch_metadata.csv"
    yolo_geojson_path = OUTPUT_DIR / "kiln_results_yolo.geojson"
    sam_geojson_path = OUTPUT_DIR / "kiln_results_sam_masks.geojson"

    st.write("📡 Connecting to Google Earth Engine...")

    with st.spinner("Processing GEE request..."):
        tile_paths, result = download_sentinel_tiles(
            [xmin, ymin, xmax, ymax],
            start_date,
            end_date,
            TILE_DIR,
            tile_size_deg=tile_size_deg,
            cloud_pct=cloud_pct,
            district_geometry=district_geometry,
            district_name=st.session_state.selected_district,
        )

    # Check if this is a batch export (large area)
    if len(tile_paths) == 1 and tile_paths[0] == "BATCH_EXPORT":
        # This is a batch export task
        instructions = result[0] if result else "Task started"
        
        st.info("📦 **Batch Export Started**")
        st.markdown(instructions)
        
        st.markdown("---")
        st.subheader("📥 Next Steps:")
        st.markdown("""
        1. **Wait 5-30 minutes** for processing to complete
        2. **Check your Google Drive** in the folder: `GEE_Brick_Kiln_Exports`
        3. **Download the GeoTIFF file** to your computer
        4. **Place the file** in the `outputs/gee_tiles/` folder
        5. **Rerun detection** with the downloaded file
        
        Alternatively, use a **smaller area** (< 500 km²) for direct processing within the app.
        """)
        
        # Offer to help with manual file placement
        if st.button("📂 Open Outputs Folder", use_container_width=True):
            os.startfile(OUTPUT_DIR)
        
        st.stop()
    
    elif len(tile_paths) == 0:
        st.error("❌ No Sentinel-2 tiles could be downloaded.")
        st.info("💡 Troubleshooting tips:")
        st.info("1. Try expanding your date range (e.g., 2023-01-01 to 2025-06-01)")
        st.info("2. Increase the Cloud Percentage Filter (try 50-80%)")
        st.info("3. Make sure your AOI is not too small")
        st.info("4. For large areas (> 500 km²), batch export will be used automatically")
        st.info("5. Click 'Test GEE Connection' button to verify GEE is working")
        st.stop()

    st.success(f"✅ Downloaded {len(tile_paths)} GeoTIFF tile(s).")

    if len(tile_paths) > 0:
        try:
            first_img, _, _ = read_tiff_as_rgb(tile_paths[0])
            st.subheader("📸 Sample Downloaded Sentinel-2 Tile")
            st.image(
                stretch_for_display(first_img),
                caption="First downloaded tile",
                use_container_width=True,
            )
        except Exception as e:
            st.warning(f"Could not display preview: {e}")

    # =========================
    # Create patches
    # =========================
    st.write("🖼️ Creating georeferenced TIFF patches...")

    total_patches = 0
    tile_patch_cache = []

    for tile_idx, tile_path in enumerate(tile_paths, start=1):
        patch_dir = PATCH_ROOT_DIR / f"tile_{tile_idx:03d}"
        patch_dir, patch_records = create_kaggle_style_tif_patches(
            tile_path,
            patch_dir,
            patch_size=PATCH_SIZE,
            overlap=OVERLAP,
            prefix=f"tile_{tile_idx:03d}"
        )

        if len(patch_records) == 0:
            st.warning(f"Tile {tile_idx} generated 0 patches and was skipped.")
            continue

        tile_patch_cache.append({
            "tile_idx": tile_idx,
            "tile_path": tile_path,
            "patch_dir": patch_dir,
            "patch_records": patch_records,
        })
        total_patches += len(patch_records)

    st.write(f"📊 Total patches created: {total_patches}")

    if total_patches == 0:
        st.error("❌ No patches created.")
        st.stop()

    st.subheader("🔍 Sample Extracted Patches")
    sample_paths = []
    for item in tile_patch_cache:
        sample_paths.extend([Path(r["patch_path"]) for r in item["patch_records"][:8]])
        if len(sample_paths) >= 8:
            break

    if sample_paths:
        sample_cols = st.columns(4)
        for i, p in enumerate(sample_paths[:8]):
            try:
                sample_img, _, _ = read_tiff_as_rgb(p)
                with sample_cols[i % 4]:
                    st.image(stretch_for_display(sample_img), caption=p.name, use_container_width=True)
            except Exception as e:
                st.warning(f"Cannot preview {p.name}: {e}")

    # =========================
    # YOLO Prediction
    # =========================
    yolo_rows = []
    obb_polygons = []
    sam_polygons = []
    patch_info = []

    raw_kiln_id = 1
    processed = 0
    progress_bar = st.progress(0)
    status_text = st.empty()

    st.write("🎯 Running YOLO-OBB prediction on saved TIFF patches...")

    for item in tile_patch_cache:
        tile_idx = item["tile_idx"]
        tile_path = item["tile_path"]
        patch_dir = item["patch_dir"]
        patch_records = item["patch_records"]
        tile_name = Path(tile_path).name

        patch_files = sorted(Path(patch_dir).glob("*.tif")) + sorted(Path(patch_dir).glob("*.tiff"))
        if len(patch_files) == 0:
            continue

        patch_record_by_name = {Path(r["patch_path"]).name: r for r in patch_records}

        yolo_results = detector.yolo.predict(
            source=str(patch_dir),
            imgsz=YOLO_IMGSZ,
            conf=conf,
            iou=0.3,
            save=False,
            save_txt=False,
            verbose=False,
        )

        for r in yolo_results:
            patch_path = Path(r.path)
            record = patch_record_by_name.get(patch_path.name, None)
            if record is None:
                continue

            processed += 1
            status_text.write(f"Processing patch {processed}/{total_patches}")

            with rasterio.open(patch_path) as src:
                transform = src.transform
                crs = src.crs
                patch_img = src.read([1, 2, 3])
                patch_img = np.transpose(patch_img, (1, 2, 0)).astype(np.uint8)

            detections_for_patch = []
            if r.obb is not None and len(r.obb) > 0:
                for obb in r.obb:
                    points = obb.xyxyxyxy.cpu().numpy()[0]
                    box_xyxy = obb.xyxy.cpu().numpy()[0]
                    confidence = float(obb.conf.cpu().numpy()[0])
                    detections_for_patch.append({
                        "confidence": confidence,
                        "obb_points": points,
                        "xyxy": box_xyxy,
                    })

            detection_count = len(detections_for_patch)
            max_conf = max([d["confidence"] for d in detections_for_patch], default=0.0)

            patch_info.append({
                "Patch_No": record["patch_no"],
                "Tile": tile_name,
                "Patch_File": patch_path.name,
                "X_pixel": record["x"],
                "Y_pixel": record["y"],
                "Detection_Count": detection_count,
                "Max_Confidence": max_conf,
            })

            patch_draw = patch_img.copy()

            sam_masks = []
            if use_sam and detection_count > 0 and detector.sam is not None:
                boxes = [d["xyxy"].tolist() for d in detections_for_patch]
                try:
                    sam_results = detector.sam.predict(
                        str(patch_path),
                        bboxes=boxes,
                        verbose=False,
                    )
                    if sam_results and sam_results[0].masks is not None:
                        sam_masks = sam_results[0].masks.data.cpu().numpy()
                except Exception as e:
                    st.warning(f"SAM failed: {e}")

            for det_idx, det in enumerate(detections_for_patch):
                points = det["obb_points"]

                geo_points = []
                for px, py in points:
                    lon, lat = xy(transform, py, px)
                    geo_points.append((lon, lat))

                obb_polygon = Polygon(geo_points)
                if not obb_polygon.is_valid or obb_polygon.area == 0:
                    continue

                temp_gdf = gpd.GeoDataFrame([{"id": raw_kiln_id}], geometry=[obb_polygon], crs=crs)
                temp_gdf_m = temp_gdf.to_crs(UTM_CRS)
                obb_area_m2 = float(temp_gdf_m.geometry.area.iloc[0])
                centroid = obb_polygon.centroid

                sam_polygon = None
                sam_area_m2 = np.nan
                if use_sam and det_idx < len(sam_masks):
                    mask_np = sam_masks[det_idx].astype(np.uint8)
                    sam_polygon, sam_area_m2 = sam_mask_to_polygon(mask_np, transform, crs)

                yolo_rows.append({
                    "Kiln_ID": f"RAW_{raw_kiln_id:05d}",
                    "Tile": tile_name,
                    "Patch_No": record["patch_no"],
                    "Patch_File": patch_path.name,
                    "Confidence": det["confidence"],
                    "Latitude": centroid.y,
                    "Longitude": centroid.x,
                    "OBB_Area_m2": obb_area_m2,
                    "SAM_Area_m2": sam_area_m2,
                })

                obb_polygons.append(obb_polygon)
                sam_polygons.append(sam_polygon)

                pts = points.astype(int)
                cv2.polylines(patch_draw, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
                cv2.putText(
                    patch_draw,
                    f"{det['confidence']:.2f}",
                    (int(points[:, 0].mean()), int(points[:, 1].mean())),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 255),
                    1,
                )

                raw_kiln_id += 1

            if detection_count > 0:
                out_patch_path = DETECTED_PATCH_DIR / f"patch_{record['patch_no']:05d}_conf_{max_conf:.2f}.jpg"
                cv2.imwrite(str(out_patch_path), cv2.cvtColor(patch_draw, cv2.COLOR_RGB2BGR))

            progress_bar.progress(processed / total_patches)

    status_text.success("✅ YOLO prediction completed!")

    patch_df = pd.DataFrame(patch_info)
    patch_df.to_csv(patch_csv_path, index=False)

    st.subheader("📋 Patch Metadata")
    st.dataframe(patch_df)

    if len(yolo_rows) == 0:
        st.warning("⚠️ No brick kilns detected in the selected AOI.")
        with open(patch_csv_path, "rb") as f:
            st.download_button("Download Patch Metadata CSV", f, file_name="patch_metadata.csv")
        st.stop()

    raw_df = pd.DataFrame(yolo_rows)
    raw_df.to_csv(raw_csv_path, index=False)

    st.write("🔄 Removing duplicate detections...")
    df, final_obb_polygons, final_sam_polygons = remove_duplicates(
        raw_df,
        obb_polygons,
        sam_polygons,
        distance_m=duplicate_distance_m,
    )

    df.to_csv(csv_path, index=False)

    st.success(f"✅ Raw YOLO detections: {len(raw_df)}")
    st.success(f"✅ Final detections after duplicate removal: {len(df)}")

    gdf_yolo = gpd.GeoDataFrame(df, geometry=final_obb_polygons, crs="EPSG:4326")
    gdf_yolo.to_file(yolo_geojson_path, driver="GeoJSON")
    _, yolo_zip_path = write_shapefile_zip(gdf_yolo, OUTPUT_DIR / "kiln_results_yolo")

    sam_zip_path = None
    if use_sam:
        sam_valid_rows = []
        sam_valid_geoms = []
        for row, sam_geom in zip(df.to_dict("records"), final_sam_polygons):
            if sam_geom is not None and not sam_geom.is_empty:
                sam_valid_rows.append(row)
                sam_valid_geoms.append(sam_geom)

        if len(sam_valid_rows) > 0:
            gdf_sam = gpd.GeoDataFrame(sam_valid_rows, geometry=sam_valid_geoms, crs="EPSG:4326")
            gdf_sam.to_file(sam_geojson_path, driver="GeoJSON")
            _, sam_zip_path = write_shapefile_zip(gdf_sam, OUTPUT_DIR / "kiln_results_sam_masks")

    all_outputs_zip = OUTPUT_DIR / "all_kiln_outputs.zip"
    with zipfile.ZipFile(all_outputs_zip, "w") as zipf:
        for file in [csv_path, patch_csv_path, raw_csv_path, yolo_geojson_path]:
            if Path(file).exists():
                zipf.write(file, arcname=Path(file).name)
        if sam_geojson_path.exists():
            zipf.write(sam_geojson_path, arcname=sam_geojson_path.name)
        for file in glob.glob(str(OUTPUT_DIR / "kiln_results_yolo.*")):
            zipf.write(file, arcname=os.path.basename(file))
        for file in glob.glob(str(OUTPUT_DIR / "kiln_results_sam_masks.*")):
            zipf.write(file, arcname=os.path.basename(file))
        for file in glob.glob(str(DETECTED_PATCH_DIR / "*.jpg"))[:50]:
            zipf.write(file, arcname=f"detected_patches/{os.path.basename(file)}")

    st.subheader("📊 Final Detected Brick Kilns")
    st.dataframe(df)

    st.subheader("🖼️ Detected Patch Samples")
    sample_patches = list(DETECTED_PATCH_DIR.glob("*.jpg"))[:8]
    if len(sample_patches) > 0:
        cols = st.columns(4)
        for i, p in enumerate(sample_patches):
            img_patch = cv2.imread(str(p))
            img_patch = cv2.cvtColor(img_patch, cv2.COLOR_BGR2RGB)
            with cols[i % 4]:
                st.image(img_patch, caption=p.name, use_container_width=True)

    st.subheader("📥 Download Results")

    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with open(csv_path, "rb") as f:
            st.download_button("📊 Final CSV", f, file_name="kiln_results.csv")
    
    with col2:
        with open(yolo_geojson_path, "rb") as f:
            st.download_button("🗺️ YOLO GeoJSON", f, file_name="kiln_results_yolo.geojson")
    
    with col3:
        with open(yolo_zip_path, "rb") as f:
            st.download_button("📦 Shapefile ZIP", f, file_name="kiln_results_yolo.zip")
    
    with col4:
        with open(all_outputs_zip, "rb") as f:
            st.download_button("💾 All Outputs ZIP", f, file_name="all_kiln_outputs.zip")

    if sam_zip_path:
        col5, col6 = st.columns(2)
        with col5:
            with open(sam_geojson_path, "rb") as f:
                st.download_button("🎭 SAM Mask GeoJSON", f, file_name="kiln_results_sam_masks.geojson")
        with col6:
            with open(sam_zip_path, "rb") as f:
                st.download_button("📦 SAM Shapefile ZIP", f, file_name="kiln_results_sam_masks.zip")

    st.success("🎉 Detection complete! Download your results above.")

st.markdown("---")
st.caption("Built with ❤️ using Google Earth Engine, YOLOv8-OBB, and SAM | Small areas: Direct download | Large areas: Google Drive export")