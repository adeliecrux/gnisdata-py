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
    use_cache: bool = False
) -> gpd.GeoDataFrame:
    """
    Download, extract, and load GNIS data into a GeoDataFrame.
    
    This function handles the entire pipeline: downloading the ZIP file,
    extracting the GPKG, and loading it into a GeoDataFrame. A temporary
    file is used for the GPKG since GeoPandas requires a file path, but
    it is automatically cleaned up.
    
    Args:
        location: Either 'National' for all US data or a two-letter state code.
                 Default is 'National'.
        layer: Specific layer name to load from the GPKG. If None, loads the
              first/default layer.
        use_cache: If True, cache the downloaded data (not implemented yet).
                  
    Returns:
        A GeoDataFrame containing the GNIS geographic names data.
        
    Raises:
        GNISDataError: If any step in the process fails.
        
    Examples:
        >>> gdf = load_gnis_gdf('CA')  # Load California data
        >>> gdf = load_gnis_gdf('National')  # Load all US data
        >>> print(gdf.head())
    """
    print(f"Downloading GNIS data for {location}...")
    zip_data = download_gnis_data(location)
    
    print("Extracting GPKG file...")
    gpkg_data = extract_gpkg_from_zip(zip_data, location)
    
    # GeoPandas requires a file path, so we need to create a temporary file
    # We use a context manager to ensure cleanup
    print("Loading data into GeoDataFrame...")
    with tempfile.NamedTemporaryFile(suffix='.gpkg', delete=True) as tmp_file:
        tmp_file.write(gpkg_data)
        tmp_file.flush()
        
        try:
            if layer:
                gdf = gpd.read_file(tmp_file.name, layer=layer)
            else:
                gdf = gpd.read_file(tmp_file.name)
                
            print(f"Successfully loaded {len(gdf)} features.")
            return gdf
            
        except Exception as e:
            raise GNISDataError(f"Failed to load GPKG into GeoDataFrame: {e}")


def get_available_states() -> set:
    """
    Get the set of valid state codes that can be used.
    
    Returns:
        A set of valid two-letter state codes.
    """
    return VALID_STATES.copy()


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
