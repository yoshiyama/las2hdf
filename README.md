# LAS-HDF5 Converter

A tool for converting point cloud data between LAS and HDF5 formats. Features high-speed conversion using PDAL and supports multiprocessing.

## Features

- Convert LAS format to HDF5 format
- Convert HDF5 format to LAS format
- Multiprocessing support
- Memory usage control
- Detailed conversion progress
- Comprehensive error output

## Requirements

- Python 3.6 or higher
- Required Python packages:
 - pdal
 - numpy
 - h5py
 - laspy
 - tqdm
 - json

## Installation
```bash
# Install required packages
pip install pdal numpy h5py laspy tqdm

# Install PDAL (for Ubuntu/WSL2)
sudo apt-get install pdal
```

## Usage
```bash
# Convert LAS to HDF5
python ST0las2hdf_perfect_kana_pdal.py input.las output.hdf5

# Convert HDF5 to LAS
python ST0las2hdf_perfect_kana_pdal.py input.hdf5 output.las --to-las
```
