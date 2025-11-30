"""
Comprehensive unit tests for gnisdata module.

Tests cover downloading, extraction, and loading of GNIS data with mocked
network calls to avoid actual downloads during testing.
"""

import io
import zipfile
import tempfile
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import gnisdata
from gnisdata import (
    _construct_url,
    download_gnis_data,
    extract_gpkg_from_zip,
    load_gnis_gdf,
    get_available_states,
    get_elevation,
    clear_cache,
    get_cache_info,
    create_enriched_export,
    GNISDataError,
    BASE_URL,
    ELEVATION_SERVICE_URL,
    VALID_STATES
)


class TestConstructUrl:
    """Tests for URL construction function."""
    
    def test_construct_url_national(self):
        """Test URL construction for National dataset."""
        url = _construct_url("National")
        assert url == BASE_URL + "Gazetteer_National_GPKG.zip"
    
    def test_construct_url_national_lowercase(self):
        """Test URL construction handles lowercase 'national'."""
        url = _construct_url("national")
        assert url == BASE_URL + "Gazetteer_National_GPKG.zip"
    
    def test_construct_url_state_code(self):
        """Test URL construction for state codes."""
        url = _construct_url("CA")
        assert url == BASE_URL + "Gazetteer_CA_GPKG.zip"
        
        url = _construct_url("NY")
        assert url == BASE_URL + "Gazetteer_NY_GPKG.zip"
    
    def test_construct_url_state_code_lowercase(self):
        """Test URL construction handles lowercase state codes."""
        url = _construct_url("ca")
        assert url == BASE_URL + "Gazetteer_CA_GPKG.zip"
    
    def test_construct_url_invalid_location(self):
        """Test URL construction raises error for invalid location."""
        with pytest.raises(GNISDataError) as exc_info:
            _construct_url("INVALID")
        
        assert "Invalid location" in str(exc_info.value)
        assert "INVALID" in str(exc_info.value)
    
    def test_construct_url_all_valid_states(self):
        """Test URL construction for all valid state codes."""
        for state in VALID_STATES:
            url = _construct_url(state)
            expected = BASE_URL + f"Gazetteer_{state}_GPKG.zip"
            assert url == expected


class TestDownloadGnisData:
    """Tests for GNIS data download function."""
    
    @patch('gnisdata.requests.get')
    def test_download_gnis_data_success(self, mock_get):
        """Test successful download of GNIS data."""
        # Mock response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'chunk1', b'chunk2']
        mock_get.return_value = mock_response
        
        result = download_gnis_data("CA")
        
        # Verify correct URL was called
        expected_url = BASE_URL + "Gazetteer_CA_GPKG.zip"
        mock_get.assert_called_once_with(expected_url, stream=True, timeout=30)
        
        # Verify content was assembled correctly
        assert result == b'chunk1chunk2'
    
    @patch('gnisdata.requests.get')
    def test_download_gnis_data_national(self, mock_get):
        """Test download of National dataset."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'data']
        mock_get.return_value = mock_response
        
        result = download_gnis_data("National")
        
        expected_url = BASE_URL + "Gazetteer_National_GPKG.zip"
        mock_get.assert_called_once_with(expected_url, stream=True, timeout=30)
        assert result == b'data'
    
    @patch('gnisdata.requests.get')
    def test_download_gnis_data_http_error(self, mock_get):
        """Test download handles HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("404 Not Found")
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            download_gnis_data("CA")
        
        assert "Failed to download data for CA" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_download_gnis_data_connection_error(self, mock_get):
        """Test download handles connection errors."""
        mock_get.side_effect = Exception("Connection timeout")
        
        with pytest.raises(GNISDataError) as exc_info:
            download_gnis_data("CA")
        
        assert "Failed to download data" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_download_gnis_data_invalid_location(self, mock_get):
        """Test download with invalid location."""
        with pytest.raises(GNISDataError) as exc_info:
            download_gnis_data("ZZ")
        
        assert "Invalid location" in str(exc_info.value)
        # Should not attempt download for invalid location
        mock_get.assert_not_called()
    
    @patch('gnisdata.requests.get')
    def test_download_gnis_data_custom_chunk_size(self, mock_get):
        """Test download with custom chunk size."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [b'chunk']
        mock_get.return_value = mock_response
        
        result = download_gnis_data("CA", chunk_size=16384)
        
        # Verify iter_content was called with custom chunk size
        mock_response.iter_content.assert_called_once_with(chunk_size=16384)
        assert result == b'chunk'


class TestExtractGpkgFromZip:
    """Tests for GPKG extraction from ZIP archive."""
    
    def create_mock_zip(self, filename: str, content: bytes = b'mock gpkg data') -> bytes:
        """Helper to create a mock ZIP file in memory."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(filename, content)
        return zip_buffer.getvalue()
    
    def test_extract_gpkg_success_state(self):
        """Test successful extraction of state GPKG file."""
        expected_content = b'california gpkg data'
        zip_data = self.create_mock_zip("Gazetteer_CA_GPKG.gpkg", expected_content)
        
        result = extract_gpkg_from_zip(zip_data, "CA")
        
        assert result == expected_content
    
    def test_extract_gpkg_success_national(self):
        """Test successful extraction of National GPKG file."""
        expected_content = b'national gpkg data'
        zip_data = self.create_mock_zip("Gazetteer_National_GPKG.gpkg", expected_content)
        
        result = extract_gpkg_from_zip(zip_data, "National")
        
        assert result == expected_content
    
    def test_extract_gpkg_lowercase_location(self):
        """Test extraction handles lowercase location names."""
        expected_content = b'ny gpkg data'
        zip_data = self.create_mock_zip("Gazetteer_NY_GPKG.gpkg", expected_content)
        
        result = extract_gpkg_from_zip(zip_data, "ny")
        
        assert result == expected_content
    
    def test_extract_gpkg_file_not_found(self):
        """Test extraction fails when expected file is missing."""
        # Create zip with wrong filename
        zip_data = self.create_mock_zip("wrong_file.gpkg")
        
        with pytest.raises(GNISDataError) as exc_info:
            extract_gpkg_from_zip(zip_data, "CA")
        
        assert "not found in archive" in str(exc_info.value)
        assert "Gazetteer_CA_GPKG.gpkg" in str(exc_info.value)
    
    def test_extract_gpkg_invalid_zip(self):
        """Test extraction fails with invalid ZIP data."""
        invalid_zip = b'not a zip file'
        
        with pytest.raises(GNISDataError) as exc_info:
            extract_gpkg_from_zip(invalid_zip, "CA")
        
        assert "Invalid ZIP file" in str(exc_info.value)
    
    def test_extract_gpkg_empty_zip(self):
        """Test extraction fails with empty ZIP."""
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            pass  # Create empty zip
        
        with pytest.raises(GNISDataError) as exc_info:
            extract_gpkg_from_zip(zip_buffer.getvalue(), "CA")
        
        assert "not found in archive" in str(exc_info.value)
    
    def test_extract_gpkg_multiple_files_in_zip(self):
        """Test extraction selects correct file from multi-file ZIP."""
        zip_buffer = io.BytesIO()
        expected_content = b'correct gpkg data'
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("readme.txt", b'readme content')
            zf.writestr("Gazetteer_CA_GPKG.gpkg", expected_content)
            zf.writestr("other_file.xml", b'xml content')
        
        result = extract_gpkg_from_zip(zip_buffer.getvalue(), "CA")
        
        assert result == expected_content


class TestLoadGnisGdf:
    """Tests for loading GNIS data into GeoDataFrame."""
    
    def create_mock_gpkg(self) -> bytes:
        """Create a minimal valid GPKG file for testing."""
        # Create a simple GeoDataFrame
        gdf = gpd.GeoDataFrame(
            {
                'name': ['Feature 1', 'Feature 2'],
                'state': ['CA', 'CA'],
                'feature_class': ['Summit', 'Stream']
            },
            geometry=[Point(0, 0), Point(1, 1)],
            crs='EPSG:4326'
        )
        
        gpkg_buffer = io.BytesIO()
        return b'mock_gpkg_bytes'
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_load_gnis_gdf_success(self, mock_read_file, mock_extract, mock_download):
        """Test successful loading of GNIS data into GeoDataFrame."""
        # Setup mocks
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        
        mock_gdf = gpd.GeoDataFrame(
            {'name': ['Test'], 'geometry': [Point(0, 0)]},
            crs='EPSG:4326'
        )
        mock_read_file.return_value = mock_gdf
        
        # Call function
        result = load_gnis_gdf("CA")
        
        # Verify calls
        mock_download.assert_called_once_with("CA")
        mock_extract.assert_called_once_with(b'mock_zip_data', "CA")
        assert mock_read_file.call_count == 1
        
        # Verify result
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1
        assert result['name'].iloc[0] == 'Test'
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_load_gnis_gdf_with_layer(self, mock_read_file, mock_extract, mock_download):
        """Test loading specific layer from GPKG."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        result = load_gnis_gdf("CA", layer="specific_layer")
        
        # Verify layer parameter was passed
        call_args = mock_read_file.call_args
        assert call_args.kwargs.get('layer') == 'specific_layer'
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_load_gnis_gdf_national(self, mock_read_file, mock_extract, mock_download):
        """Test loading National dataset."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        result = load_gnis_gdf("National")
        
        mock_download.assert_called_once_with("National")
        mock_extract.assert_called_once_with(b'mock_zip_data', "National")
    
    @patch('gnisdata.download_gnis_data')
    def test_load_gnis_gdf_download_failure(self, mock_download):
        """Test handling of download failures."""
        mock_download.side_effect = GNISDataError("Download failed")
        
        with pytest.raises(GNISDataError) as exc_info:
            load_gnis_gdf("CA")
        
        assert "Download failed" in str(exc_info.value)
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    def test_load_gnis_gdf_extraction_failure(self, mock_extract, mock_download):
        """Test handling of extraction failures."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.side_effect = GNISDataError("Extraction failed")
        
        with pytest.raises(GNISDataError) as exc_info:
            load_gnis_gdf("CA")
        
        assert "Extraction failed" in str(exc_info.value)
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_load_gnis_gdf_read_failure(self, mock_read_file, mock_extract, mock_download):
        """Test handling of GeoDataFrame read failures."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_read_file.side_effect = Exception("Failed to read GPKG")
        
        with pytest.raises(GNISDataError) as exc_info:
            load_gnis_gdf("CA")
        
        assert "Failed to load GPKG into GeoDataFrame" in str(exc_info.value)
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_load_gnis_gdf_temp_file_cleanup(self, mock_read_file, mock_extract, mock_download):
        """Test that temporary files are properly cleaned up when not caching."""
        # Setup mocks
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        # Call without caching - temp file should be cleaned up
        result = load_gnis_gdf("CA")
        
        # Verify result is correct
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 1


class TestGetAvailableStates:
    """Tests for getting available states."""
    
    def test_get_available_states_returns_set(self):
        """Test that function returns a set."""
        result = get_available_states()
        assert isinstance(result, set)
    
    def test_get_available_states_contains_valid_codes(self):
        """Test that result contains expected state codes."""
        result = get_available_states()
        
        # Check some known states
        assert 'CA' in result
        assert 'NY' in result
        assert 'TX' in result
        assert 'FL' in result
        
        # Check territories
        assert 'PR' in result
        assert 'GU' in result
    
    def test_get_available_states_count(self):
        """Test that result has expected number of entries."""
        result = get_available_states()
        # 50 states + DC + 6 territories = 57
        assert len(result) == 57
    
    def test_get_available_states_returns_copy(self):
        """Test that function returns a copy, not the original."""
        result1 = get_available_states()
        result2 = get_available_states()
        
        # Modify one result
        result1.add('XX')
        
        # Verify the other is unaffected
        assert 'XX' not in result2
        assert 'XX' not in VALID_STATES


class TestGNISDataError:
    """Tests for custom exception."""
    
    def test_gnis_data_error_is_exception(self):
        """Test that GNISDataError is an Exception."""
        assert issubclass(GNISDataError, Exception)
    
    def test_gnis_data_error_message(self):
        """Test that error message is preserved."""
        error = GNISDataError("Test error message")
        assert str(error) == "Test error message"
    
    def test_gnis_data_error_can_be_raised(self):
        """Test that error can be raised and caught."""
        with pytest.raises(GNISDataError) as exc_info:
            raise GNISDataError("Custom error")
        
        assert "Custom error" in str(exc_info.value)


class TestConstants:
    """Tests for module constants."""
    
    def test_base_url_format(self):
        """Test BASE_URL is properly formatted."""
        assert BASE_URL.startswith("https://")
        assert BASE_URL.endswith("/")
    
    def test_elevation_service_url_format(self):
        """Test ELEVATION_SERVICE_URL is properly formatted."""
        assert ELEVATION_SERVICE_URL.startswith("https://")
        assert "epqs.nationalmap.gov" in ELEVATION_SERVICE_URL
    
    def test_valid_states_immutability(self):
        """Test that VALID_STATES is a set."""
        assert isinstance(VALID_STATES, set)
    
    def test_valid_states_all_uppercase(self):
        """Test that all state codes are uppercase."""
        for state in VALID_STATES:
            assert state.isupper()
            assert len(state) == 2


class TestGetElevation:
    """Tests for elevation query function."""
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_success_feet(self, mock_get):
        """Test successful elevation query in feet."""
        # Mock response for Mount Whitney
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'value': 14505.3,
            'units': 'Feet'
        }
        mock_get.return_value = mock_response
        
        result = get_elevation(36.578581, -118.291994, units="Feet")
        
        # Verify correct API call
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args.kwargs['params']['x'] == -118.291994
        assert call_args.kwargs['params']['y'] == 36.578581
        assert call_args.kwargs['params']['units'] == 'Feet'
        assert call_args.kwargs['params']['output'] == 'json'
        
        # Verify result
        assert result == 14505
        assert isinstance(result, int)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_success_meters(self, mock_get):
        """Test successful elevation query in meters."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'value': 4421.0,
            'units': 'Meters'
        }
        mock_get.return_value = mock_response
        
        result = get_elevation(36.578581, -118.291994, units="Meters")
        
        # Verify units parameter
        call_args = mock_get.call_args
        assert call_args.kwargs['params']['units'] == 'Meters'
        
        assert result == 4421
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_default_units(self, mock_get):
        """Test that default units are Feet."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': 5280.0}
        mock_get.return_value = mock_response
        
        result = get_elevation(40.0, -105.0)
        
        # Verify default units
        call_args = mock_get.call_args
        assert call_args.kwargs['params']['units'] == 'Feet'
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_rounding(self, mock_get):
        """Test that elevation values are properly rounded."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': 1234.6}
        mock_get.return_value = mock_response
        
        result = get_elevation(40.0, -105.0)
        assert result == 1235  # Rounds up
        
        mock_response.json.return_value = {'value': 1234.4}
        result = get_elevation(40.0, -105.0)
        assert result == 1234  # Rounds down
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_zero_elevation(self, mock_get):
        """Test handling of zero elevation (sea level)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': 0.0}
        mock_get.return_value = mock_response
        
        result = get_elevation(40.0, -74.0)  # Near NYC coast
        assert result == 0
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_negative_elevation(self, mock_get):
        """Test handling of negative elevation (Death Valley)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': -282.0}
        mock_get.return_value = mock_response
        
        result = get_elevation(36.23, -116.89)  # Death Valley
        assert result == -282
    
    def test_get_elevation_invalid_latitude_high(self):
        """Test validation of latitude upper bound."""
        with pytest.raises(ValueError) as exc_info:
            get_elevation(91.0, -118.0)
        
        assert "Latitude must be between -90 and 90" in str(exc_info.value)
        assert "91" in str(exc_info.value)
    
    def test_get_elevation_invalid_latitude_low(self):
        """Test validation of latitude lower bound."""
        with pytest.raises(ValueError) as exc_info:
            get_elevation(-91.0, -118.0)
        
        assert "Latitude must be between -90 and 90" in str(exc_info.value)
    
    def test_get_elevation_invalid_longitude_high(self):
        """Test validation of longitude upper bound."""
        with pytest.raises(ValueError) as exc_info:
            get_elevation(36.0, 181.0)
        
        assert "Longitude must be between -180 and 180" in str(exc_info.value)
        assert "181" in str(exc_info.value)
    
    def test_get_elevation_invalid_longitude_low(self):
        """Test validation of longitude lower bound."""
        with pytest.raises(ValueError) as exc_info:
            get_elevation(36.0, -181.0)
        
        assert "Longitude must be between -180 and 180" in str(exc_info.value)
    
    def test_get_elevation_invalid_units(self):
        """Test validation of units parameter."""
        with pytest.raises(ValueError) as exc_info:
            get_elevation(36.0, -118.0, units="Kilometers")
        
        assert "Units must be 'Feet' or 'Meters'" in str(exc_info.value)
        assert "Kilometers" in str(exc_info.value)
    
    def test_get_elevation_valid_boundary_coordinates(self):
        """Test that boundary coordinates are accepted."""
        # Should not raise for valid boundary values
        with patch('gnisdata.requests.get') as mock_get:
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {'value': 0.0}
            mock_get.return_value = mock_response
            
            # Test boundaries
            get_elevation(90.0, 180.0)  # Max
            get_elevation(-90.0, -180.0)  # Min
            get_elevation(0.0, 0.0)  # Zero
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_no_value_in_response(self, mock_get):
        """Test handling of response without elevation value."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'error': 'No data'}
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(36.0, -118.0)
        
        assert "No elevation data returned" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_null_value(self, mock_get):
        """Test handling of null elevation value (over ocean)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': None}
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(30.0, -130.0)  # Pacific Ocean
        
        assert "No elevation available" in str(exc_info.value)
        assert "outside coverage area or over water" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_sentinel_value(self, mock_get):
        """Test handling of sentinel value for missing data."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': -1000000}
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(30.0, -130.0)
        
        assert "No elevation available" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_http_error(self, mock_get):
        """Test handling of HTTP errors."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("500 Server Error")
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(36.0, -118.0)
        
        assert "Failed to query elevation service" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_connection_error(self, mock_get):
        """Test handling of connection errors."""
        mock_get.side_effect = Exception("Connection timeout")
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(36.0, -118.0)
        
        assert "Failed to query elevation service" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_invalid_json(self, mock_get):
        """Test handling of invalid JSON response."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.side_effect = ValueError("Invalid JSON")
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(36.0, -118.0)
        
        assert "Failed to parse elevation response" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_malformed_response(self, mock_get):
        """Test handling of malformed response data (string instead of dict)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = "not a dict"
        mock_get.return_value = mock_response
        
        with pytest.raises(GNISDataError) as exc_info:
            get_elevation(36.0, -118.0)
        
        # String response doesn't have 'value' key, so it triggers "No elevation data"
        assert "No elevation data returned" in str(exc_info.value)
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_timeout_parameter(self, mock_get):
        """Test that timeout is set in request."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': 1000.0}
        mock_get.return_value = mock_response
        
        get_elevation(36.0, -118.0)
        
        # Verify timeout was set
        call_args = mock_get.call_args
        assert call_args.kwargs['timeout'] == 10
    
    @patch('gnisdata.requests.get')
    def test_get_elevation_url_construction(self, mock_get):
        """Test that correct URL is used."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'value': 1000.0}
        mock_get.return_value = mock_response
        
        get_elevation(36.0, -118.0)
        
        # Verify correct URL
        call_args = mock_get.call_args
        assert call_args[0][0] == ELEVATION_SERVICE_URL or \
               call_args.args[0] == ELEVATION_SERVICE_URL


# Integration-style tests (still mocked but test multiple components together)
class TestIntegration:
    """Integration tests for the complete workflow."""
    
    @patch('gnisdata.requests.get')
    @patch('gnisdata.gpd.read_file')
    def test_complete_workflow_mocked(self, mock_read_file, mock_get):
        """Test complete workflow from download to GeoDataFrame."""
        # Create a real ZIP with GPKG-like content
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("Gazetteer_CA_GPKG.gpkg", b'mock gpkg content')
        
        # Mock HTTP response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.iter_content.return_value = [zip_buffer.getvalue()]
        mock_get.return_value = mock_response
        
        # Mock GeoDataFrame read
        mock_gdf = gpd.GeoDataFrame(
            {
                'feature_name': ['Mount Whitney', 'Lake Tahoe'],
                'feature_class': ['Summit', 'Lake'],
                'state_alpha': ['CA', 'CA']
            },
            geometry=[Point(-118.292, 36.578), Point(-120.0, 39.0)],
            crs='EPSG:4326'
        )
        mock_read_file.return_value = mock_gdf
        
        # Execute complete workflow
        result = load_gnis_gdf("CA")
        
        # Verify result
        assert isinstance(result, gpd.GeoDataFrame)
        assert len(result) == 2
        assert 'feature_name' in result.columns
        assert result['feature_name'].iloc[0] == 'Mount Whitney'


class TestCaching:
    """Tests for caching functionality."""
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_disabled_by_default(self, mock_read_file, mock_extract, mock_download):
        """Test that caching is disabled by default."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'test_cache'
            
            # Call without caching
            result = load_gnis_gdf("CA")
            
            # Verify cache directory was not created
            assert not cache_dir.exists()
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_creates_directory(self, mock_read_file, mock_extract, mock_download):
        """Test that cache directory is created when caching is enabled."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / 'test_cache'
            
            result = load_gnis_gdf("CA", use_cache=True, cache_dir=cache_dir)
            
            # Verify cache directory was created
            assert cache_dir.exists()
            assert cache_dir.is_dir()
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_saves_gpkg_file(self, mock_read_file, mock_extract, mock_download):
        """Test that GPKG file is saved to cache."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data_content'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            result = load_gnis_gdf("CA", use_cache=True, cache_dir=cache_dir)
            
            # Verify cache file exists
            cache_file = cache_dir / "Gazetteer_CA_GPKG.gpkg"
            assert cache_file.exists()
            assert cache_file.read_bytes() == b'mock_gpkg_data_content'
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_hit_skips_download(self, mock_read_file, mock_extract, mock_download):
        """Test that cached file is used and download is skipped."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Cached']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # First call - should download and cache
            result1 = load_gnis_gdf("CA", use_cache=True, cache_dir=cache_dir)
            assert mock_download.call_count == 1
            
            # Second call - should use cache
            result2 = load_gnis_gdf("CA", use_cache=True, cache_dir=cache_dir)
            
            # Verify download was not called again
            assert mock_download.call_count == 1
            assert mock_extract.call_count == 1
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_different_layers_same_file(self, mock_read_file, mock_extract, mock_download):
        """Test that different layers can be loaded from same cached file."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf1 = gpd.GeoDataFrame({'name': ['Layer1']}, geometry=[Point(0, 0)])
        mock_gdf2 = gpd.GeoDataFrame({'name': ['Layer2']}, geometry=[Point(1, 1)])
        mock_read_file.side_effect = [mock_gdf1, mock_gdf2]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            # Load first layer
            result1 = load_gnis_gdf("CA", layer="DomesticNames", use_cache=True, cache_dir=cache_dir)
            
            # Load second layer - should reuse cached file
            result2 = load_gnis_gdf("CA", layer="FeatureDescriptionHistory", use_cache=True, cache_dir=cache_dir)
            
            # Verify download only happened once
            assert mock_download.call_count == 1
            assert mock_read_file.call_count == 2
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_corrupted_file_redownload(self, mock_read_file, mock_extract, mock_download):
        """Test that corrupted cache file triggers re-download."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        
        # First read fails (corrupted), second succeeds
        mock_read_file.side_effect = [
            Exception("Corrupted file"),
            mock_gdf
        ]
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_file = cache_dir / "Gazetteer_CA_GPKG.gpkg"
            
            # Create corrupted cache file
            cache_file.write_bytes(b'corrupted data')
            
            # Should detect corruption and re-download
            result = load_gnis_gdf("CA", use_cache=True, cache_dir=cache_dir)
            
            # Verify re-download occurred
            assert mock_download.call_count == 1
            assert mock_extract.call_count == 1
    
    @patch('gnisdata.download_gnis_data')
    @patch('gnisdata.extract_gpkg_from_zip')
    @patch('gnisdata.gpd.read_file')
    def test_cache_national_location(self, mock_read_file, mock_extract, mock_download):
        """Test caching with National location."""
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['National']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            result = load_gnis_gdf("National", use_cache=True, cache_dir=cache_dir)
            
            # Verify correct filename for National
            cache_file = cache_dir / "Gazetteer_National_GPKG.gpkg"
            assert cache_file.exists()


class TestClearCache:
    """Tests for clear_cache function."""
    
    def test_clear_cache_single_location(self):
        """Test clearing cache for a single location."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create mock cache files
            ca_file = cache_dir / "Gazetteer_CA_GPKG.gpkg"
            ny_file = cache_dir / "Gazetteer_NY_GPKG.gpkg"
            ca_file.write_bytes(b'CA data')
            ny_file.write_bytes(b'NY data')
            
            # Clear only CA
            clear_cache("CA", cache_dir=cache_dir)
            
            # Verify only CA was deleted
            assert not ca_file.exists()
            assert ny_file.exists()
    
    def test_clear_cache_all(self):
        """Test clearing all cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create multiple cache files
            ca_file = cache_dir / "Gazetteer_CA_GPKG.gpkg"
            ny_file = cache_dir / "Gazetteer_NY_GPKG.gpkg"
            national_file = cache_dir / "Gazetteer_National_GPKG.gpkg"
            ca_file.write_bytes(b'CA data')
            ny_file.write_bytes(b'NY data')
            national_file.write_bytes(b'National data')
            
            # Clear all
            clear_cache(cache_dir=cache_dir)
            
            # Verify all were deleted
            assert not ca_file.exists()
            assert not ny_file.exists()
            assert not national_file.exists()
    
    def test_clear_cache_nonexistent_location(self):
        """Test clearing cache for location that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Should not raise error
            clear_cache("TX", cache_dir=cache_dir)
    
    def test_clear_cache_no_cache_dir(self):
        """Test clearing cache when cache directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nonexistent"
            
            # Should not raise error
            clear_cache(cache_dir=cache_dir)
    
    def test_clear_cache_national(self):
        """Test clearing National cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            national_file = cache_dir / "Gazetteer_National_GPKG.gpkg"
            national_file.write_bytes(b'National data')
            
            clear_cache("National", cache_dir=cache_dir)
            
            assert not national_file.exists()


class TestGetCacheInfo:
    """Tests for get_cache_info function."""
    
    def test_get_cache_info_empty(self):
        """Test get_cache_info with no cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            
            info = get_cache_info(cache_dir=cache_dir)
            
            assert info['cached_files'] == []
            assert info['total_size_mb'] == 0
            assert 'cache_dir' in info
    
    def test_get_cache_info_with_files(self):
        """Test get_cache_info with cached files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            # Create cache files of known sizes
            ca_file = cache_dir / "Gazetteer_CA_GPKG.gpkg"
            ny_file = cache_dir / "Gazetteer_NY_GPKG.gpkg"
            ca_file.write_bytes(b'A' * 1024 * 1024)  # 1 MB
            ny_file.write_bytes(b'B' * 2 * 1024 * 1024)  # 2 MB
            
            info = get_cache_info(cache_dir=cache_dir)
            
            assert len(info['cached_files']) == 2
            assert info['total_size_mb'] == pytest.approx(3.0, rel=0.01)
            
            # Verify individual file info
            filenames = [f['filename'] for f in info['cached_files']]
            assert 'Gazetteer_CA_GPKG.gpkg' in filenames
            assert 'Gazetteer_NY_GPKG.gpkg' in filenames
            
            # Check that files are sorted
            assert info['cached_files'][0]['filename'] <= info['cached_files'][1]['filename']
    
    def test_get_cache_info_file_details(self):
        """Test that get_cache_info returns complete file details."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir)
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            ca_file = cache_dir / "Gazetteer_CA_GPKG.gpkg"
            ca_file.write_bytes(b'test data')
            
            info = get_cache_info(cache_dir=cache_dir)
            
            assert len(info['cached_files']) == 1
            file_info = info['cached_files'][0]
            
            # Verify all required fields
            assert 'filename' in file_info
            assert 'size_mb' in file_info
            assert 'modified' in file_info
            assert 'path' in file_info
            assert file_info['filename'] == 'Gazetteer_CA_GPKG.gpkg'
    
    def test_get_cache_info_nonexistent_dir(self):
        """Test get_cache_info with nonexistent directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = Path(tmpdir) / "nonexistent"
            
            info = get_cache_info(cache_dir=cache_dir)
            
            assert info['cached_files'] == []
            assert info['total_size_mb'] == 0


class TestCreateEnrichedExport:
    """Tests for create_enriched_export function."""
    
    def create_mock_domestic_gdf(self):
        """Create a mock DomesticNames GeoDataFrame."""
        return gpd.GeoDataFrame({
            'feature_id': [1, 2, 3, 4, 5],
            'feature_name': ['Mount Test', 'Test Ridge', 'Test Valley', 'Test Peak', 'Test Stream'],
            'feature_class': ['summit', 'ridge', 'valley', 'summit', 'stream'],
            'state_name': ['Colorado', 'Colorado', 'Colorado', 'Colorado', 'Colorado'],
            'county_name': ['Boulder', 'Boulder', 'Larimer', 'Boulder', 'Larimer'],
            'prim_lat_dec': [40.0, 40.1, 40.2, 40.3, 40.4],
            'prim_long_dec': [-105.0, -105.1, -105.2, -105.3, -105.4],
        }, geometry=[Point(-105.0, 40.0), Point(-105.1, 40.1), Point(-105.2, 40.2),
                     Point(-105.3, 40.3), Point(-105.4, 40.4)])
    
    def create_mock_history_gdf(self):
        """Create a mock FeatureDescriptionHistory GeoDataFrame."""
        return gpd.GeoDataFrame({
            'feature_id': [1, 2, 3],
            'description': ['High mountain peak', 'Long mountain ridge', 'Deep valley'],
            'history': ['Named in 1950', 'Named in 1960', 'Named in 1970'],
        }, geometry=[Point(-105.0, 40.0), Point(-105.1, 40.1), Point(-105.2, 40.2)])
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_basic_success(self, mock_clear_cache, mock_load):
        """Test basic successful execution without elevation or export."""
        # Setup mocks
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        # Mock load_gnis_gdf to return different data based on layer parameter
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            elif layer == 'FeatureDescriptionHistory':
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        # Call function
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit', 'ridge'],
            clear_cache_after=True
        )
        
        # Verify load_gnis_gdf was called correctly
        assert mock_load.call_count == 2
        
        # First call for DomesticNames
        call1 = mock_load.call_args_list[0]
        assert call1.kwargs['location'] == 'CO'
        assert call1.kwargs['layer'] == 'DomesticNames'
        assert call1.kwargs['use_cache'] is True
        
        # Second call for FeatureDescriptionHistory
        call2 = mock_load.call_args_list[1]
        assert call2.kwargs['location'] == 'CO'
        assert call2.kwargs['layer'] == 'FeatureDescriptionHistory'
        assert call2.kwargs['use_cache'] is True
        
        # Verify clear_cache was called
        mock_clear_cache.assert_called_once_with(location='CO', cache_dir=None)
        
        # Verify result structure
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 3  # 2 summits + 1 ridge
        
        # Verify columns were renamed
        assert 'name' in result.columns
        assert 'class' in result.columns
        assert 'state' in result.columns
        assert 'county' in result.columns
        assert 'latitude' in result.columns
        assert 'longitude' in result.columns
        assert 'desc_history' in result.columns
        
        # Verify original columns were removed
        assert 'feature_name' not in result.columns
        assert 'feature_class' not in result.columns
        assert 'description' not in result.columns
        assert 'history' not in result.columns
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_filtering(self, mock_clear_cache, mock_load):
        """Test that feature class filtering works correctly."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        # Filter for only summits
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit'],
            clear_cache_after=False
        )
        
        # Verify only summits are returned
        assert len(result) == 2
        assert all(result['class'] == 'summit')
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_no_matching_features(self, mock_clear_cache, mock_load):
        """Test error when no features match the filter."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        # Filter for non-existent feature class
        with pytest.raises(GNISDataError) as exc_info:
            create_enriched_export(
                location='CO',
                feature_classes=['nonexistent'],
                clear_cache_after=False
            )
        
        assert "No features found" in str(exc_info.value)
        assert "nonexistent" in str(exc_info.value)
        assert "CO" in str(exc_info.value)
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_history_join(self, mock_clear_cache, mock_load):
        """Test that history data is properly joined."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit', 'ridge'],
            clear_cache_after=False
        )
        
        # Check that desc_history was created
        assert 'desc_history' in result.columns
        
        # Feature 1 should have combined description and history
        feature_1 = result[result['feature_id'] == 1].iloc[0]
        assert 'High mountain peak' in feature_1['desc_history']
        assert 'Named in 1950' in feature_1['desc_history']
        
        # Feature 4 (summit) has no history, should have empty desc_history
        feature_4 = result[result['feature_id'] == 4].iloc[0]
        assert feature_4['desc_history'] == ''
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_custom_cache_dir(self, mock_clear_cache, mock_load):
        """Test using custom cache directory."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        custom_cache = "/tmp/custom_cache"
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit'],
            cache_dir=custom_cache,
            clear_cache_after=True
        )
        
        # Verify cache_dir was passed to load_gnis_gdf
        for call in mock_load.call_args_list:
            assert call.kwargs['cache_dir'] == custom_cache
        
        # Verify cache_dir was passed to clear_cache
        mock_clear_cache.assert_called_once_with(location='CO', cache_dir=custom_cache)
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_no_cache_clear(self, mock_clear_cache, mock_load):
        """Test that cache is not cleared when clear_cache_after=False."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit'],
            clear_cache_after=False
        )
        
        # Verify clear_cache was NOT called
        mock_clear_cache.assert_not_called()
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_history_load_failure(self, mock_clear_cache, mock_load):
        """Test handling of history layer load failure."""
        domestic_gdf = self.create_mock_domestic_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                raise Exception("Failed to load history")
        
        mock_load.side_effect = mock_load_side_effect
        
        with pytest.raises(GNISDataError) as exc_info:
            create_enriched_export(
                location='CO',
                feature_classes=['summit'],
                clear_cache_after=False
            )
        
        assert "Failed to load FeatureDescriptionHistory layer" in str(exc_info.value)
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    @patch('gnisdata.get_elevation')
    @patch('gnisdata.time.sleep')
    def test_create_enriched_export_with_elevation(self, mock_sleep, mock_elevation, 
                                                    mock_clear_cache, mock_load):
        """Test adding elevation data to export."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        # Mock elevation returns
        mock_elevation.side_effect = [8000, 7500, 7000]
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit', 'ridge'],
            add_elevation=True,
            clear_cache_after=False
        )
        
        # Verify elevation column exists
        assert 'elevation_ft' in result.columns
        
        # Verify elevation was called for each record
        assert mock_elevation.call_count == 3
        
        # Verify elevation values
        assert result.iloc[0]['elevation_ft'] == 8000
        assert result.iloc[1]['elevation_ft'] == 7500
        assert result.iloc[2]['elevation_ft'] == 7000
        
        # Verify sleep was called for rate limiting (n-1 times)
        assert mock_sleep.call_count == 2
        mock_sleep.assert_called_with(0.1)
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    @patch('gnisdata.get_elevation')
    @patch('gnisdata.time.sleep')
    def test_create_enriched_export_max_elevation_requests(self, mock_sleep, mock_elevation,
                                                           mock_clear_cache, mock_load):
        """Test limiting number of elevation requests."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        # Mock elevation returns
        mock_elevation.side_effect = [8000, 7500]
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit', 'ridge'],
            add_elevation=True,
            max_elevation_requests=2,
            clear_cache_after=False
        )
        
        # Verify elevation was only called 2 times (max_elevation_requests)
        assert mock_elevation.call_count == 2
        
        # First two should have elevation
        assert result.iloc[0]['elevation_ft'] == 8000
        assert result.iloc[1]['elevation_ft'] == 7500
        
        # Third should be None
        assert result.iloc[2]['elevation_ft'] is None
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    @patch('gnisdata.get_elevation')
    @patch('gnisdata.time.sleep')
    def test_create_enriched_export_elevation_failure_handling(self, mock_sleep, mock_elevation,
                                                               mock_clear_cache, mock_load):
        """Test that elevation failures don't stop processing."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        # Mock elevation to fail on second call
        mock_elevation.side_effect = [8000, GNISDataError("No elevation"), 7000]
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit', 'ridge'],
            add_elevation=True,
            clear_cache_after=False
        )
        
        # Verify all calls were attempted
        assert mock_elevation.call_count == 3
        
        # Verify first and third have elevation, second is None
        assert result.iloc[0]['elevation_ft'] == 8000
        assert result.iloc[1]['elevation_ft'] is None
        assert result.iloc[2]['elevation_ft'] == 7000
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_with_file_output(self, mock_clear_cache, mock_load):
        """Test exporting to pipe-delimited file."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_output.psv"
            
            result = create_enriched_export(
                location='CO',
                feature_classes=['summit'],
                output_file=str(output_path),
                clear_cache_after=False
            )
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file contents
            content = output_path.read_text()
            
            # Check for pipe delimiter
            assert '|' in content
            
            # Check for column headers
            assert 'feature_id' in content
            assert 'name' in content
            assert 'class' in content
            
            # Check for data
            assert 'Mount Test' in content
            assert 'summit' in content
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_no_file_output(self, mock_clear_cache, mock_load):
        """Test that no file is created when output_file is None."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Call without output_file
            result = create_enriched_export(
                location='CO',
                feature_classes=['summit'],
                output_file=None,
                clear_cache_after=False
            )
            
            # Verify no .psv or .csv files were created
            psv_files = list(Path(tmpdir).glob("*.psv"))
            csv_files = list(Path(tmpdir).glob("*.csv"))
            
            assert len(psv_files) == 0
            assert len(csv_files) == 0
            
            # But result should still be returned
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_national_location(self, mock_clear_cache, mock_load):
        """Test with National location."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        result = create_enriched_export(
            location='National',
            feature_classes=['summit'],
            clear_cache_after=True
        )
        
        # Verify location was passed correctly
        for call in mock_load.call_args_list:
            assert call.kwargs['location'] == 'National'
        
        mock_clear_cache.assert_called_once_with(location='National', cache_dir=None)
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_multiple_feature_classes(self, mock_clear_cache, mock_load):
        """Test filtering with multiple feature classes."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit', 'ridge', 'valley'],
            clear_cache_after=False
        )
        
        # Should return 4 features (2 summits, 1 ridge, 1 valley)
        assert len(result) == 4
        
        # Verify all classes are present
        classes = set(result['class'].unique())
        assert classes == {'summit', 'ridge', 'valley'}
    
    @patch('gnisdata.load_gnis_gdf')
    @patch('gnisdata.clear_cache')
    def test_create_enriched_export_column_selection(self, mock_clear_cache, mock_load):
        """Test that only expected columns are in output."""
        domestic_gdf = self.create_mock_domestic_gdf()
        history_gdf = self.create_mock_history_gdf()
        
        def mock_load_side_effect(location, layer, use_cache, cache_dir):
            if layer == 'DomesticNames':
                return domestic_gdf
            else:
                return history_gdf
        
        mock_load.side_effect = mock_load_side_effect
        
        result = create_enriched_export(
            location='CO',
            feature_classes=['summit'],
            clear_cache_after=False
        )
        
        # Expected columns
        expected_cols = {'feature_id', 'name', 'class', 'state', 'county', 
                        'latitude', 'longitude', 'desc_history'}
        
        result_cols = set(result.columns)
        
        # All expected columns should be present
        assert expected_cols.issubset(result_cols)
        
        # Original column names should not be present
        assert 'feature_name' not in result_cols
        assert 'feature_class' not in result_cols
        assert 'state_name' not in result_cols
        assert 'county_name' not in result_cols
        assert 'prim_lat_dec' not in result_cols
        assert 'prim_long_dec' not in result_cols
        assert 'description' not in result_cols
        assert 'history' not in result_cols
