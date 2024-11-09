\# LAS-HDF5 Converter

A Python program for converting point cloud data between LAS and HDF5 formats. The program utilizes PDAL for high-speed conversion and supports multiprocessing capabilities.

\## Features

\* LAS to HDF5 conversion
\* HDF5 to LAS conversion
\* Multiprocessing support
\* Memory usage control
\* Progress tracking
\* Detailed error reporting

\## Requirements

\* Python 3.6 or higher
\* Required Python packages:
  \* pdal
  \* numpy
  \* h5py
  \* laspy
  \* tqdm
  \* json

\## Installation

1. Install required packages:
\```bash
pip install pdal numpy h5py laspy tqdm
\```

2. Install PDAL:
   \* Ubuntu:
     \```bash
     sudo apt-get install pdal
     \```
   \* macOS:
     \```bash
     brew install pdal
     \```
