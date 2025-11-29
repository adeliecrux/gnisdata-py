"""
Comprehensive unit tests for gnisdata module.

Tests cover downloading, extraction, and loading of GNIS data with mocked
network calls to avoid actual downloads during testing.
"""

import io
import zipfile
import pytest
from unittest.mock import Mock, patch, MagicMock
import geopandas as gpd
from shapely.geometry import Point

import gnisdata
from gnisdata import (
    _construct_url,
    download_gnis_data,
    extract_gpkg_from_zip,
    load_gnis_gdf,
    get_available_states,
    GNISDataError,
    BASE_URL,
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
        
        # Write to in-memory file
        gpkg_buffer = io.BytesIO()
        # Note: GeoPandas can't write to BytesIO directly for GPKG, 
        # so we'll mock this in the test
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
    @patch('gnisdata.tempfile.NamedTemporaryFile')
    def test_load_gnis_gdf_temp_file_cleanup(self, mock_tempfile, mock_read_file, 
                                             mock_extract, mock_download):
        """Test that temporary files are properly cleaned up."""
        # Setup mocks
        mock_download.return_value = b'mock_zip_data'
        mock_extract.return_value = b'mock_gpkg_data'
        mock_gdf = gpd.GeoDataFrame({'name': ['Test']}, geometry=[Point(0, 0)])
        mock_read_file.return_value = mock_gdf
        
        # Mock the context manager for NamedTemporaryFile
        mock_tmp = MagicMock()
        mock_tmp.name = '/tmp/mock_file.gpkg'
        mock_tmp.__enter__.return_value = mock_tmp
        mock_tempfile.return_value = mock_tmp
        
        result = load_gnis_gdf("CA")
        
        # Verify temp file was written
        assert mock_tmp.write.called
        assert mock_tmp.flush.called
        
        # Verify context manager was used (ensures cleanup)
        assert mock_tmp.__enter__.called
        assert mock_tmp.__exit__.called


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
    
    def test_valid_states_immutability(self):
        """Test that VALID_STATES is a set."""
        assert isinstance(VALID_STATES, set)
    
    def test_valid_states_all_uppercase(self):
        """Test that all state codes are uppercase."""
        for state in VALID_STATES:
            assert state.isupper()
            assert len(state) == 2


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
