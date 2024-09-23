# DELTA_iDISCO

**DELTA_iDISCO** is a pipeline designed for processing and analyzing imaging data from iDISCO experiments, specifically for registering brain images to the Allen Brain Atlas and extracting regional statistics for further analysis.

This pipeline is based on methods and resources from the following work:

Pisano, T. J. (2019). Connectivity of the Posterior Cerebellum: Transsynaptic Viral Tracing with Light-Sheet Imaged Whole-Mouse Brains. [https://dataspace.princeton.edu/handle/88435/dsp01jq085n77b](https://dataspace.princeton.edu/handle/88435/dsp01jq085n77b)

Source code and paramater are taken form here [Dennis lab](https://github.com/the-dennis-lab/cleared_brains/)


## Setup Instructions

### Using Conda

1. Install the Conda environment:
    `conda env create -f environment.yml`

2. Activate the environment:
    `conda activate delta_idisco`

### Using Pip

1. Install the necessary dependencies:
    `pip install -r requirements.txt`

## How to Use

### Single Animal Processing

To process a single animal using the pipeline, look at `main.ipynb` notebook

### Batch Processing on a Cluster

To process multiple animals in parallel on a high-performance cluster, use the `submit_jobs.sh` script. This will submit a separate job for each animal using `bsub`.

1. Edit the `submit_jobs.sh` script to specify the animal IDs and the base directory for the data.
2. Submit the jobs:
    `cd scripts`
    
    `bash submit_jobs.sh`

This will:
- Submit a job for each animal using the LSF job scheduler.
- Process the images for each animal in parallel on the cluster.

## Example Results

After processing, you can expect the following outputs:
- Transformed images for each channel (`ch0`, `ch1`, `ch2`) saved in the output directory for each animal.
- A CSV file (`region_stats.csv`) containing region-based statistics (mean intensity, area, etc.) for each animal.

### Example of `region_stats.csv`:

| Region | Mean_ch0 | Mean_ch1 | Mean_ch2 | N   |
|--------|----------|----------|----------|-----|
| 1      | 1234.56  | 2345.67  | 3456.78  | 100 |
| 2      | 1456.78  | 2678.90  | 3678.90  | 150 |

## Contributing

We welcome contributions to improve the DELTA_iDISCO pipeline. Please follow these steps:

1. Fork the repository.
2. Create a feature branch: `git checkout -b feature-branch-name`
3. Commit your changes: `git commit -m "Add new feature"`
4. Push to the branch: `git push origin feature-branch-name`
5. Open a Pull Request.

## License

This project is licensed under the MIT License.
