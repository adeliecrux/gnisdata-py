"""
gnisdata - A Python package for downloading and processing USGS GNIS data.

This module provides functionality to download, extract, and load Geographic Names
Information System (GNIS) data from the USGS into GeoPandas GeoDataFrames.
"""

import io
import zipfile
import tempfile
from pathlib import Path
from typing import Optional, Union
import requests
import geopandas as gpd


# Constants
BASE_URL = "https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/FullModel/"
ELEVATION_SERVICE_URL = "https://epqs.nationalmap.gov/v1/json"
VALID_STATES = {
    'AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA',
    'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD',
    'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ',
    'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC',
    'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY',
    'DC', 'AS', 'GU', 'MP', 'PR', 'VI', 'UM', 'DC'
}
VALID_ALL_LOCATIONS = {'NATIONAL', 'ALL', 'US', 'USA'}


class GNISDataError(Exception):
    """Base exception for GNIS data operations."""
    pass


def _construct_url(location: str) -> str:
    """
    Construct the download URL for a specific location.
    
    Args:
        location: Either a two-letter state code or one of 'National', 'All', 'US', 'USA' for all data.
        
    Returns:
        The complete URL for downloading the data.
        
    Raises:
        GNISDataError: If the location is invalid.
    """
    location = location.upper()
    
    if location in VALID_ALL_LOCATIONS:
        filename = "Gazetteer_National_GPKG.zip"
    elif location in VALID_STATES:
        filename = f"Gazetteer_{location}_GPKG.zip"
    else:
        raise GNISDataError(
            f"Invalid location: {location}. Must be 'National' or a valid state code."
        )
    
    return BASE_URL + filename


def download_gnis_data(location: str = "National", chunk_size: int = 8192) -> bytes:
    """
    Download GNIS data from USGS for a specified location.
    
    Args:
        location: Either 'National' for all US data or a two-letter state code
                 (e.g., 'CA', 'NY', 'TX'). Default is 'National'.
        chunk_size: Size of chunks to download in bytes. Default is 8192.
        
    Returns:
        The downloaded ZIP file content as bytes.
        
    Raises:
        GNISDataError: If download fails or location is invalid.
        
    Examples:
        >>> data = download_gnis_data('CA')  # Download California data
        >>> data = download_gnis_data('National')  # Download all US data
    """
    url = _construct_url(location)
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Download in chunks to handle large files efficiently
        zip_content = io.BytesIO()
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                zip_content.write(chunk)
        
        zip_content.seek(0)
        return zip_content.getvalue()
        
    except Exception as e:
        raise GNISDataError(f"Failed to download data for {location}: {e}")


def extract_gpkg_from_zip(zip_data: bytes, location: str = "National") -> bytes:
    """
    Extract the .gpkg file from the ZIP archive.
    
    Args:
        zip_data: The ZIP file content as bytes.
        location: The location identifier to construct the expected filename.
                 
    Returns:
        The extracted .gpkg file content as bytes.
        
    Raises:
        GNISDataError: If extraction fails or the expected file is not found.
    """
    location = location.upper()
    
    if location == 'NATIONAL':
        expected_filename = "Gazetteer_National_GPKG.gpkg"
    else:
        expected_filename = f"Gazetteer_{location}_GPKG.gpkg"
    
    try:
        with zipfile.ZipFile(io.BytesIO(zip_data)) as zf:
            # Check if the expected file exists
            if expected_filename not in zf.namelist():
                raise GNISDataError(
                    f"Expected file '{expected_filename}' not found in archive. "
                    f"Available files: {zf.namelist()}"
                )
            
            # Extract the GPKG file
            gpkg_content = zf.read(expected_filename)
            return gpkg_content
            
    except zipfile.BadZipFile as e:
        raise GNISDataError(f"Invalid ZIP file: {e}")


def load_gnis_gdf(
    location: str = "National",
    layer: Optional[str] = None,
    use_cache: bool = False,
    cache_dir: Optional[Union[str, Path]] = None
) -> gpd.GeoDataFrame:
    """
    Download, extract, and load GNIS data into a GeoDataFrame.
    
    This function handles the entire pipeline: downloading the ZIP file,
    extracting the GPKG, and loading it into a GeoDataFrame. When caching is
    enabled, the GPKG file is stored on disk for reuse across sessions and
    enables loading different layers without re-downloading.
    
    Args:
        location: Either 'National' for all US data or a two-letter state code.
                 Default is 'National'.
        layer: Specific layer name to load from the GPKG. If None, loads the
              first/default layer.
        use_cache: If True, cache the GPKG file for reuse. Default is False.
        cache_dir: Directory to store cached files. If None, uses ~/.cache/gnisdata.
                  Only used when use_cache=True.
                  
    Returns:
        A GeoDataFrame containing the GNIS geographic names data.
        
    Raises:
        GNISDataError: If any step in the process fails.
        
    Examples:
        >>> gdf = load_gnis_gdf('CA')  # Load California data
        >>> gdf = load_gnis_gdf('CA', use_cache=True)  # Cache for reuse
        >>> gdf = load_gnis_gdf('CA', layer='DomesticNames', use_cache=True)  # Different layer from cache
    """
    location_upper = location.upper()
    
    # Determine GPKG filename
    if location_upper in VALID_ALL_LOCATIONS:
        gpkg_filename = "Gazetteer_National_GPKG.gpkg"
    else:
        gpkg_filename = f"Gazetteer_{location_upper}_GPKG.gpkg"
    
    # Set up cache path if caching is enabled
    if use_cache:
        if cache_dir is None:
            cache_path = Path.home() / '.cache' / 'gnisdata'
        else:
            cache_path = Path(cache_dir)
        
        cache_path.mkdir(parents=True, exist_ok=True)
        gpkg_file = cache_path / gpkg_filename
        
        # Check if cached file exists
        if gpkg_file.exists():
            print(f"Using cached GPKG: {gpkg_file}")
            try:
                if layer:
                    gdf = gpd.read_file(gpkg_file, layer=layer)
                else:
                    gdf = gpd.read_file(gpkg_file)
                print(f"Successfully loaded {len(gdf)} features from cache.")
                return gdf
            except Exception as e:
                print(f"Warning: Cached file corrupted, re-downloading... ({e})")
                gpkg_file.unlink()  # Delete corrupted cache
    
    # Download and extract
    print(f"Downloading GNIS data for {location}...")
    zip_data = download_gnis_data(location)
    
    print("Extracting GPKG file...")
    gpkg_data = extract_gpkg_from_zip(zip_data, location)
    
    # Save to cache or temp file
    if use_cache:
        print(f"Caching GPKG to {gpkg_file}...")
        gpkg_file.write_bytes(gpkg_data)
        file_to_read = gpkg_file
        cleanup_needed = False
    else:
        # Use temporary file (original behavior)
        tmp_file = tempfile.NamedTemporaryFile(suffix='.gpkg', delete=False)
        tmp_file.write(gpkg_data)
        tmp_file.close()
        file_to_read = Path(tmp_file.name)
        cleanup_needed = True
    
    # Load into GeoDataFrame
    print("Loading data into GeoDataFrame...")
    try:
        if layer:
            gdf = gpd.read_file(file_to_read, layer=layer)
        else:
            gdf = gpd.read_file(file_to_read)
        
        print(f"Successfully loaded {len(gdf)} features.")
        return gdf
    
    except Exception as e:
        raise GNISDataError(f"Failed to load GPKG into GeoDataFrame: {e}")
    
    finally:
        # Clean up temp file if not using cache
        if cleanup_needed:
            file_to_read.unlink(missing_ok=True)


def get_available_states() -> set:
    """
    Get the set of valid state codes that can be used.
    
    Returns:
        A set of valid two-letter state codes.
    """
    return VALID_STATES.copy()


def clear_cache(
    location: Optional[str] = None,
    cache_dir: Optional[Union[str, Path]] = None
) -> None:
    """
    Clear cached GPKG files.
    
    Args:
        location: If specified, only clear cache for this location.
                 If None, clear all cached files.
        cache_dir: Cache directory. If None, uses ~/.cache/gnisdata
        
    Examples:
        >>> clear_cache('CA')  # Clear only California cache
        >>> clear_cache()  # Clear all cached files
    """
    if cache_dir is None:
        cache_path = Path.home() / '.cache' / 'gnisdata'
    else:
        cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        print("No cache directory found.")
        return
    
    if location:
        location_upper = location.upper()
        if location_upper in VALID_ALL_LOCATIONS:
            filename = "Gazetteer_National_GPKG.gpkg"
        else:
            filename = f"Gazetteer_{location_upper}_GPKG.gpkg"
        
        file_to_delete = cache_path / filename
        if file_to_delete.exists():
            file_to_delete.unlink()
            print(f"Cleared cache for {location}")
        else:
            print(f"No cache found for {location}")
    else:
        # Clear all
        count = 0
        for gpkg in cache_path.glob("*.gpkg"):
            gpkg.unlink()
            count += 1
        if count > 0:
            print(f"Cleared {count} cached GPKG file(s)")
        else:
            print("No cached files to clear")


def get_cache_info(cache_dir: Optional[Union[str, Path]] = None) -> dict:
    """
    Get information about cached files.
    
    Args:
        cache_dir: Cache directory. If None, uses ~/.cache/gnisdata
    
    Returns:
        Dictionary with cache information including:
        - cache_dir: Path to cache directory
        - cached_files: List of cached files with size and modification time
        - total_size_mb: Total size of all cached files in megabytes
        
    Examples:
        >>> info = get_cache_info()
        >>> print(f"Total cache size: {info['total_size_mb']:.2f} MB")
        >>> for file in info['cached_files']:
        ...     print(f"{file['filename']}: {file['size_mb']:.2f} MB")
    """
    if cache_dir is None:
        cache_path = Path.home() / '.cache' / 'gnisdata'
    else:
        cache_path = Path(cache_dir)
    
    if not cache_path.exists():
        return {
            'cache_dir': str(cache_path),
            'cached_files': [],
            'total_size_mb': 0
        }
    
    cached_files = []
    total_size = 0
    
    for gpkg in sorted(cache_path.glob("*.gpkg")):
        size = gpkg.stat().st_size
        total_size += size
        cached_files.append({
            'filename': gpkg.name,
            'size_mb': size / (1024 * 1024),
            'modified': gpkg.stat().st_mtime,
            'path': str(gpkg)
        })
    
    return {
        'cache_dir': str(cache_path),
        'cached_files': cached_files,
        'total_size_mb': total_size / (1024 * 1024)
    }


def get_elevation(latitude: float, longitude: float, units: str = "Feet") -> int:
    """
    Get elevation for a specific latitude/longitude using the USGS Elevation Point Query Service.
    
    This function queries the USGS National Map Elevation Point Query Service (EPQS)
    to retrieve elevation data for a given coordinate.
    
    Args:
        latitude: Decimal latitude (e.g., 36.578581)
        longitude: Decimal longitude (e.g., -118.291994)
        units: Units for elevation. Either "Feet" or "Meters". Default is "Feet".
        
    Returns:
        Elevation as an integer in the specified units.
        
    Raises:
        GNISDataError: If the API request fails or returns invalid data.
        ValueError: If latitude/longitude are out of valid range or units are invalid.
        
    Examples:
        >>> elevation = get_elevation(36.578581, -118.291994)  # Mount Whitney
        >>> print(f"Elevation: {elevation} feet")
        Elevation: 14505 feet
        
        >>> elevation = get_elevation(36.578581, -118.291994, units="Meters")
        >>> print(f"Elevation: {elevation} meters")
        Elevation: 4421 meters
    """
    # Validate inputs
    if not -90 <= latitude <= 90:
        raise ValueError(f"Latitude must be between -90 and 90, got {latitude}")
    
    if not -180 <= longitude <= 180:
        raise ValueError(f"Longitude must be between -180 and 180, got {longitude}")
    
    if units not in ("Feet", "Meters"):
        raise ValueError(f"Units must be 'Feet' or 'Meters', got {units}")
    
    # Construct API request
    params = {
        'x': longitude,
        'y': latitude,
        'units': units,
        'output': 'json'
    }
    
    try:
        response = requests.get(ELEVATION_SERVICE_URL, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Check if elevation data is present
        if 'value' not in data:
            raise GNISDataError(
                f"No elevation data returned for coordinates ({latitude}, {longitude}). "
                f"Response: {data}"
            )
        
        elevation_value = data['value']
        
        # Handle case where elevation is not available (e.g., over ocean)
        if elevation_value is None or elevation_value == -1000000:
            raise GNISDataError(
                f"No elevation available for coordinates ({latitude}, {longitude}). "
                "Location may be outside coverage area or over water."
            )
        
        # Return as integer
        return int(round(elevation_value))
        
    except GNISDataError:
        # Re-raise GNISDataError as-is
        raise
    except requests.exceptions.RequestException as e:
        raise GNISDataError(f"Failed to query elevation service: {e}")
    except (KeyError, ValueError, TypeError) as e:
        raise GNISDataError(f"Failed to parse elevation response: {e}")
    except Exception as e:
        raise GNISDataError(f"Failed to query elevation service: {e}")


if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) > 1:
        location = sys.argv[1]
    else:
        location = "National"
        
    try:
        gdf = load_gnis_gdf(location)
        print(f"\nLoaded {len(gdf)} features for {location}")
        print(f"\nColumns: {list(gdf.columns)}")
        print(f"\nFirst 5 rows:")
        print(gdf.head())
        print(f"\nGeometry type: {gdf.geom_type.unique()}")
        
    except GNISDataError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


