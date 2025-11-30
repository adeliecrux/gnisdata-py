# gnisdata-py

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A Python package for downloading and processing USGS Geographic Names Information System (GNIS) data. Easily access official geographic names data for all US states and territories with built-in caching and optional elevation enrichment.

## Features

- 🗺️ **Easy data access** - Download GNIS data for any US state or nationwide
- 💾 **Smart caching** - Optional file-based caching for faster repeated access
- 📊 **Multiple layers** - Load different GPKG layers (DomesticNames, FeatureDescriptionHistory, etc.)
- 🏔️ **Elevation enrichment** - Query USGS Elevation Point Query Service (EPQS) API
- 🔄 **Advanced workflows** - Combine multiple layers with filtering and enrichment
- 🐼 **Pandas/GeoPandas integration** - Returns familiar DataFrame/GeoDataFrame objects
- ✅ **Well-tested** - 90 unit tests with 80% code coverage

## Installation

```bash
pip install gnisdata
```

Or with Poetry:

```bash
poetry add gnisdata
```

## Quick Start

### Basic Usage

```python
from gnisdata import load_gnis_gdf

# Load California geographic names data
gdf = load_gnis_gdf('CA')
print(f"Loaded {len(gdf)} features")
print(gdf.head())

# Enable caching for faster repeated access
gdf = load_gnis_gdf('CA', use_cache=True)

# Load a specific layer
gdf = load_gnis_gdf('CA', layer='DomesticNames', use_cache=True)
```

### Enriched Export Workflow

Combine multiple GPKG layers, filter by feature classes, and optionally add elevation data:

```python
from gnisdata import create_enriched_export

# Get all summits in Colorado with description/history
df = create_enriched_export(
    location='CO',
    feature_classes=['Summit'],
    clear_cache_after=False
)

# Multiple feature classes with elevation (first 100 records)
df = create_enriched_export(
    location='CO',
    feature_classes=['Summit', 'Ridge', 'Valley'],
    add_elevation=True,
    max_elevation_requests=100,
    output_file='colorado_features.psv'
)
```

### Get Elevation Data

```python
from gnisdata import get_elevation

# Mount Whitney elevation
elevation = get_elevation(36.578581, -118.291994)
print(f"Elevation: {elevation} feet")  # 14505 feet

# Get elevation in meters
elevation_m = get_elevation(36.578581, -118.291994, units="Meters")
print(f"Elevation: {elevation_m} meters")  # 4421 meters
```

### Cache Management

```python
from gnisdata import get_cache_info, clear_cache

# Get cache information
info = get_cache_info()
print(f"Cache directory: {info['cache_dir']}")
print(f"Total size: {info['total_size_mb']:.2f} MB")
for file in info['cached_files']:
    print(f"  {file['filename']}: {file['size_mb']:.2f} MB")

# Clear cache for specific location
clear_cache('CA')

# Clear all cached files
clear_cache()
```

## Command-Line Interface

The package includes a CLI for quick data exploration:

```bash
# Show help and usage examples
python -m gnisdata

# Load and explore Colorado data
python -m gnisdata CO

# Create enriched export for summits
python -m gnisdata CO Summit

# Multiple feature classes
python -m gnisdata CO Summit,Ridge,Valley
```

## Supported Locations

All 50 US states, DC, and territories:
- States: AL, AK, AZ, AR, CA, CO, CT, DE, FL, GA, HI, ID, IL, IN, IA, KS, KY, LA, ME, MD, MA, MI, MN, MS, MO, MT, NE, NV, NH, NJ, NM, NY, NC, ND, OH, OK, OR, PA, RI, SC, SD, TN, TX, UT, VT, VA, WA, WV, WI, WY
- Other: DC, AS, GU, MP, PR, VI, UM
- National: Use 'National' for nationwide data

## Available GPKG Layers

Common layers in GNIS GPKG files:
- `DomesticNames` - Primary geographic names
- `FeatureDescriptionHistory` - Feature descriptions and historical notes
- `FederalCodes` - Federal agency codes
- `NationalFedCodes` - National federal codes
- And more (use QGIS or `fiona.listlayers()` to explore)

## Common Feature Classes

Examples of feature classes available in the data:
- `Summit` - Mountain peaks and summits
- `Ridge` - Mountain ridges
- `Valley` - Valleys
- `Stream` - Rivers and streams
- `Lake` - Lakes and ponds
- `Bay` - Bays and coves
- `Cape` - Capes and points
- `Island` - Islands
- And many more...

[Reference PDF](https://prd-tnm.s3.amazonaws.com/StagedProducts/GeographicNames/GNIS_file_format.pdf)

## Error Handling

The package uses a custom `GNISDataError` exception for all GNIS-specific errors:

```python
from gnisdata import load_gnis_gdf, GNISDataError

try:
    gdf = load_gnis_gdf('INVALID')
except GNISDataError as e:
    print(f"Error: {e}")
```

## Development

### Setup

```bash
# Clone repository
git clone https://github.com/adeliecrux/gnisdata-py.git
cd gnisdata-py

# Install with Poetry
poetry install

# Run tests
poetry run pytest

# Run tests with coverage
poetry run pytest --cov=gnisdata --cov-report=term-missing
```

### Running Tests

```bash
# All tests
poetry run pytest

# Specific test class
poetry run pytest tests/test_gnisdata.py::TestCreateEnrichedExport

# With verbose output
poetry run pytest -v

# With coverage report
poetry run pytest --cov=gnisdata --cov-report=html
```

### Code Quality

```bash
# Format code
poetry run black gnisdata.py tests/

# Sort imports
poetry run isort gnisdata.py tests/

# Lint
poetry run flake8 gnisdata.py tests/
```

## Requirements

- Python 3.10+
- geopandas >= 1.1.1
- requests >= 2.32.5
- pandas (installed with geopandas)

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run the test suite (`poetry run pytest`)
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Citation

If you use this package in research, please cite:

```
gnisdata-py: A Python package for USGS GNIS data access and processing
URL: https://github.com/adeliecrux/gnisdata-py
```

## Acknowledgments

- Data provided by the [U.S. Geological Survey](https://www.usgs.gov/)
    - **GNIS Data**: [USGS Geographic Names](https://www.usgs.gov/u.s.-board-on-geographic-names/download-gnis-data)
    - **Elevation Data**: [USGS Elevation Point Query Service (EPQS)](https://apps.nationalmap.gov/epqs/)
- Geographic Names Information System (GNIS) maintained by the [U.S. Board on Geographic Names](https://www.usgs.gov/u.s.-board-on-geographic-names)
This package downloads data from:

## Support

- **Issues**: [GitHub Issues](https://github.com/adeliecrux/gnisdata-py/issues)
- **Documentation**: This README
- **Email**: chad.m.kahl@gmail.com
