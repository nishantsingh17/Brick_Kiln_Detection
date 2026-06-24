from pathlib import Path

import ee
import requests
import geopandas as gpd
from shapely.geometry import mapping
from shapely.ops import transform
import pyproj
from shapely.geometry import Polygon, MultiPolygon


def initialize_gee():
    """Initialize Google Earth Engine"""
    MY_PROJECT_ID = "brick-kiln-detection-498810"
    try:
        ee.Initialize(project=MY_PROJECT_ID)
        print("✅ GEE Initialized successfully")
        return True
    except Exception as e:
        print(f"⚠️ Authentication needed: {e}")
        try:
            ee.Authenticate()
            ee.Initialize(project=MY_PROJECT_ID)
            print("✅ GEE Initialized after authentication")
            return True
        except Exception as auth_error:
            print(f"❌ GEE Authentication failed: {auth_error}")
            return False


def test_gee_connection():
    """Test if GEE is working and data is available"""
    try:
        if not initialize_gee():
            return False
        
        test_point = ee.Geometry.Point([79.0455, 26.7157])
        
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(test_point)
            .filterDate("2023-01-01", "2025-06-01")
        )
        
        count = collection.size().getInfo()
        print(f"✅ GEE connection successful! Found {count} Sentinel-2 images")
        return count > 0
        
    except Exception as e:
        print(f"❌ GEE connection failed: {e}")
        return False


def split_polygon_into_grid(geometry, grid_size_km=10):
    """
    Split a large polygon into smaller grid cells for easier downloading
    """
    # Convert to meters for grid splitting
    project = pyproj.Transformer.from_crs('EPSG:4326', 'EPSG:32644', always_xy=True).transform
    project_back = pyproj.Transformer.from_crs('EPSG:32644', 'EPSG:4326', always_xy=True).transform
    
    # Transform polygon to UTM (meters)
    geom_utm = transform(project, geometry)
    
    # Get bounds in meters
    minx, miny, maxx, maxy = geom_utm.bounds
    grid_size_m = grid_size_km * 1000
    
    # Create grid cells
    cells = []
    x = minx
    while x < maxx:
        y = miny
        while y < maxy:
            x2 = min(x + grid_size_m, maxx)
            y2 = min(y + grid_size_m, maxy)
            
            # Create cell polygon in UTM
            cell_polygon = Polygon([
                (x, y), (x2, y), (x2, y2), (x, y2), (x, y)
            ])
            
            # Transform back to WGS84
            cell_wgs84 = transform(project_back, cell_polygon)
            
            # Check if intersects with original geometry
            if cell_wgs84.intersects(geometry):
                cells.append(cell_wgs84.intersection(geometry))
            
            y = y2
        x = x2
    
    return cells


def download_single_tile(aoi_geometry, start_date, end_date, out_path, cloud_pct=40):
    """
    Download a single tile (for both small and large areas)
    """
    try:
        # Convert shapely geometry to ee geometry
        if isinstance(aoi_geometry, (Polygon, MultiPolygon)):
            coords = list(aoi_geometry.exterior.coords) if hasattr(aoi_geometry, 'exterior') else list(aoi_geometry.geoms[0].exterior.coords)
            aoi = ee.Geometry.Polygon(coords)
        else:
            # Bounding box
            xmin, ymin, xmax, ymax = aoi_geometry
            aoi = ee.Geometry.Rectangle([xmin, ymin, xmax, ymax])

        # Check if there are any images
        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", cloud_pct))
        )
        
        count = collection.size().getInfo()
        if count == 0:
            print(f"No images found for this tile")
            return None

        # Get median composite
        image = collection.median().select(["B4", "B3", "B2"]).clip(aoi)
        
        # Get download URL
        url = image.getDownloadURL({
            "scale": 10,
            "region": aoi,
            "format": "GEO_TIFF",
            "crs": "EPSG:4326",
        })

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"Downloading to: {out_path.name}")
        r = requests.get(url, timeout=180)
        r.raise_for_status()
        
        with open(out_path, "wb") as f:
            f.write(r.content)
        
        print(f"✅ Downloaded: {out_path.name}")
        return str(out_path)
        
    except Exception as e:
        print(f"Error downloading tile: {e}")
        return None


def download_sentinel_tiles(aoi_bounds, start_date, end_date, out_dir, tile_size_deg=0.08, cloud_pct=40, district_geometry=None, district_name=None):
    """
    Smart download - automatically handles both small and large areas
    """
    print("\n" + "="*60)
    print("Starting Sentinel-2 download")
    print("="*60)
    
    if not initialize_gee():
        return [], []

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    tile_paths = []
    
    if district_geometry is not None:
        # Using district boundary
        bounds = district_geometry.bounds
        width_km = (bounds[2] - bounds[0]) * 111
        height_km = (bounds[3] - bounds[1]) * 111
        area_km2 = width_km * height_km
        
        print(f"District: {district_name}")
        print(f"Area: {area_km2:.1f} km²")
        
        # Calculate grid size based on area
        if area_km2 > 2000:
            grid_size = 15  # 15km grid for very large districts
            print(f"Large district - splitting into {grid_size}km grid cells")
        elif area_km2 > 500:
            grid_size = 10  # 10km grid for medium districts
            print(f"Medium district - splitting into {grid_size}km grid cells")
        else:
            grid_size = None  # Download whole district
            print(f"Small district - downloading whole area")
        
        if grid_size is not None:
            # Split district into grid
            print(f"Splitting district into {grid_size}km grid cells...")
            grid_cells = split_polygon_into_grid(district_geometry, grid_size_km=grid_size)
            print(f"Created {len(grid_cells)} grid cells")
            
            # Download each grid cell
            for i, cell in enumerate(grid_cells, 1):
                print(f"\n--- Downloading cell {i}/{len(grid_cells)} ---")
                out_path = out_dir / f"tile_{i:03d}.tif"
                result = download_single_tile(cell, start_date, end_date, out_path, cloud_pct)
                if result:
                    tile_paths.append(result)
        else:
            # Download whole district
            out_path = out_dir / "district_full.tif"
            result = download_single_tile(district_geometry, start_date, end_date, out_path, cloud_pct)
            if result:
                tile_paths.append(result)
    
    else:
        # Rectangle mode - use traditional splitting
        width_km = (aoi_bounds[2] - aoi_bounds[0]) * 111
        height_km = (aoi_bounds[3] - aoi_bounds[1]) * 111
        area_km2 = width_km * height_km
        
        print(f"Custom rectangle area: {area_km2:.1f} km²")
        
        if area_km2 > 500:
            # Split rectangle into grid
            rect_polygon = Polygon([
                (aoi_bounds[0], aoi_bounds[1]),
                (aoi_bounds[2], aoi_bounds[1]),
                (aoi_bounds[2], aoi_bounds[3]),
                (aoi_bounds[0], aoi_bounds[3])
            ])
            grid_size = 10
            print(f"Splitting rectangle into {grid_size}km grid cells...")
            grid_cells = split_polygon_into_grid(rect_polygon, grid_size_km=grid_size)
            print(f"Created {len(grid_cells)} grid cells")
            
            for i, cell in enumerate(grid_cells, 1):
                print(f"\n--- Downloading cell {i}/{len(grid_cells)} ---")
                out_path = out_dir / f"tile_{i:03d}.tif"
                result = download_single_tile(cell, start_date, end_date, out_path, cloud_pct)
                if result:
                    tile_paths.append(result)
        else:
            # Download whole rectangle
            out_path = out_dir / "rectangle.tif"
            result = download_single_tile(aoi_bounds, start_date, end_date, out_path, cloud_pct)
            if result:
                tile_paths.append(result)

    print("\n" + "="*60)
    print(f"Download complete: {len(tile_paths)} tiles downloaded")
    print("="*60)
    
    return tile_paths, []