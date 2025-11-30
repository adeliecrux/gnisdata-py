# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.1] - 2025-11-30

### Added
- GitHub Actions workflow for automated PyPI publishing on releases
- Pull request checks workflow with comprehensive testing and validation
- Flake8 badge to README
- `.flake8` configuration file for linting standards

### Changed
- Version Bump

## [0.2.0] - 2025-11-30

### Added
- Initial release of gnisdata-py package
- `load_gnis_gdf()` - Download and load GNIS data into GeoDataFrame
- `download_gnis_data()` - Download GNIS ZIP files from USGS S3
- `extract_gpkg_from_zip()` - Extract GPKG files from ZIP archives
- `get_elevation()` - Query USGS Elevation Point Query Service (EPQS) API
- `create_enriched_export()` - Combine DomesticNames and FeatureDescriptionHistory layers with optional elevation enrichment
- `get_available_states()` - Get list of valid state codes
- `clear_cache()` - Clear cached GPKG files
- `get_cache_info()` - Get information about cached files
- File-based caching system at `~/.cache/gnisdata` for persistent GPKG storage
- Support for all 50 US states, DC, and territories
- Support for multiple GPKG layers (DomesticNames, FeatureDescriptionHistory, etc.)
- Command-line interface for quick data exploration
- Custom `GNISDataError` exception for error handling
- Comprehensive test suite with 90 unit tests and 80% code coverage
- Full documentation in README with examples and API reference
- Code quality tools: black, flake8, isort
- Poetry-based dependency management
- MIT License