import argparse
import os
import numpy as np
import h5py
import pdal
import json
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import time
import laspy

def convert_las_to_hdf5(las_file_path, hdf5_file_path):
    print(f"Converting {las_file_path} to {hdf5_file_path}...")
    try:
        start_time = time.time()

        # Step 1: Create PDAL pipeline
        print("Step 1: Creating PDAL pipeline...")
        step1_start = time.time()
        pipeline = pdal.Pipeline(json.dumps([
            {
                "type": "readers.las",
                "filename": las_file_path
            }
        ]))
        step1_time = time.time() - step1_start
        print(f"Step 1 completed in {step1_time:.2f} seconds")

        # Step 2: Execute pipeline
        print("Step 2: Executing PDAL pipeline...")
        step2_start = time.time()
        pipeline.execute()
        step2_time = time.time() - step2_start
        print(f"Step 2 completed in {step2_time:.2f} seconds")

        # Step 3: Get point view
        print("Step 3: Getting point view...")
        step3_start = time.time()
        print("Step 3.1: Getting point view...")
        view = pipeline.arrays[0]
        print("Step 3.2: Getting point view...")
        step3_time = time.time() - step3_start
        print(f"Step 3 completed in {step3_time:.2f} seconds")

        # Step 4: Get dimensions
        print("Step 4: Getting dimensions...")
        step4_start = time.time()
        dimensions = view.dtype.names
        step4_time = time.time() - step4_start
        print(f"Step 4 completed in {step4_time:.2f} seconds")

        # Step 5: Write data to HDF5 file
        print("Step 5: Writing data to HDF5 file...")
        step5_start = time.time()
        with h5py.File(hdf5_file_path, 'w') as hdf:
            for dim in dimensions:
                chunk_size = min(len(view[dim]), 1000000)  # Adjust chunk size as needed
                hdf.create_dataset(dim,
                                   data=view[dim],
                                   chunks=(chunk_size,),
                                   compression='gzip',
                                   compression_opts=4,
                                   shuffle=True)

            hdf.create_dataset('metadata', data=json.dumps(pipeline.metadata))
        step5_time = time.time() - step5_start
        print(f"Step 5 completed in {step5_time:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Conversion completed for {las_file_path} in {total_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error converting {las_file_path}: {str(e)}")
        return False


def convert_hdf5_to_las(hdf5_file_path, las_file_path):
    print(f"Converting {hdf5_file_path} to {las_file_path}...")
    try:
        start_time = time.time()

        # Step 1: Read HDF5 file
        print("Step 1: Reading HDF5 file...")
        step1_start = time.time()
        with h5py.File(hdf5_file_path, 'r') as hdf:
            data = {key: np.array(hdf[key][:]) for key in hdf.keys() if key != 'metadata'}
            metadata = json.loads(hdf['metadata'][()]) if 'metadata' in hdf else {}
        step1_time = time.time() - step1_start
        print(f"Step 1 completed in {step1_time:.2f} seconds")

        # Step 2: Create LAS file
        print("Step 2: Creating LAS file...")
        step2_start = time.time()

        header = laspy.LasHeader(point_format=3, version="1.2")
        las = laspy.LasData(header)

        # Required fields
        las.x = data['X']
        las.y = data['Y']
        las.z = data['Z']

        # Optional fields
        if 'Intensity' in data:
            las.intensity = data['Intensity']
        if 'ReturnNumber' in data:
            las.return_number = data['ReturnNumber']
        if 'NumberOfReturns' in data:
            las.number_of_returns = data['NumberOfReturns']
        if 'ScanDirectionFlag' in data:
            las.scan_direction_flag = data['ScanDirectionFlag']
        if 'EdgeOfFlightLine' in data:
            las.edge_of_flight_line = data['EdgeOfFlightLine']
        if 'Classification' in data:
            las.classification = data['Classification']
        if 'ScanAngleRank' in data:
            las.scan_angle_rank = data['ScanAngleRank']
        if 'UserData' in data:
            las.user_data = data['UserData']
        if 'PointSourceId' in data:
            las.point_source_id = data['PointSourceId']

        # Color information
        if 'Red' in data and 'Green' in data and 'Blue' in data:
            las.red = data['Red']
            las.green = data['Green']
            las.blue = data['Blue']

        las.write(las_file_path)

        step2_time = time.time() - step2_start
        print(f"Step 2 completed in {step2_time:.2f} seconds")

        total_time = time.time() - start_time
        print(f"Conversion completed for {hdf5_file_path} in {total_time:.2f} seconds")
        return True
    except Exception as e:
        print(f"Error converting {hdf5_file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def convert_single_file(file_pair):
    input_file, output_file = file_pair
    if input_file.lower().endswith('.las') and output_file.lower().endswith('.hdf5'):
        return convert_las_to_hdf5(input_file, output_file)
    elif input_file.lower().endswith('.hdf5') and output_file.lower().endswith('.las'):
        return convert_hdf5_to_las(input_file, output_file)
    else:
        print(f"Unsupported file conversion: {input_file} to {output_file}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Convert between LAS and HDF5 files using PDAL.")
    parser.add_argument("input_file", help="Input file path")
    parser.add_argument("output_file", help="Output file path")
    parser.add_argument("--to-las", action="store_true", help="Convert HDF5 to LAS (default is LAS to HDF5)")
    parser.add_argument("--num-processes", type=int, default=max(1, cpu_count() - 1),
                        help="Number of processes to use (default: number of CPU cores minus 1)")
    parser.add_argument("--max-memory", type=float, default=None,
                        help="Maximum memory to use in GB. If specified, it will limit the number of processes.")
    args = parser.parse_args()

    # Adjust number of processes based on max memory if specified
    if args.max_memory is not None:
        mem_per_process = 2  # Estimated memory usage per process in GB
        max_processes = int(args.max_memory / mem_per_process)
        args.num_processes = min(args.num_processes, max_processes)
        print(f"Limiting to {args.num_processes} processes based on {args.max_memory}GB max memory.")

    print(f"Using {args.num_processes} processes for conversion.")

    # Perform conversion
    if args.to_las:
        result = convert_hdf5_to_las(args.input_file, args.output_file)
    else:
        result = convert_las_to_hdf5(args.input_file, args.output_file)

    # Print summary
    if result:
        print("Conversion completed successfully.")
    else:
        print("Conversion failed.")

if __name__ == "__main__":
    main()