# DELTA_iDISCO

**DELTA_iDISCO** is a pipeline designed for processing and analyzing imaging data from iDISCO experiments, specifically for registering brain images to the Allen Brain Atlas and extracting regional statistics for further analysis. It assumes a directory structure and file names that are the default output of the [MuVi](https://www.bruker.com/en/products-and-solutions/fluorescence-microscopy/light-sheet-microscopes/muvi-spim-family.html) system.

Please change the functions in utils.py for different inputs.

This pipeline is based on methods and resources from the following work:

Pisano, T. J. (2019). Connectivity of the Posterior Cerebellum: Transsynaptic Viral Tracing with Light-Sheet Imaged Whole-Mouse Brains. [Thesis](https://dataspace.princeton.edu/handle/88435/dsp01jq085n77b)

Source code and paramaters are mostly copied from the following Dennis lab [repo](https://github.com/the-dennis-lab/cleared_brains/)

The atlas is fromn the paper cited below, cropped to fit our hemi-brains:
Perens, J., Salinas, C. G., Skytte, J. L., Roostalu, U., Dahl, A. B., Dyrby, T. B., Wichern, F., Barkholt, P., Vrang, N., Jelsing, J., & Hecksher-Sørensen, J. (2021). An Optimized Mouse Brain Atlas for Automated Mapping and Quantification of Neuronal Activity Using iDISCO+ and Light Sheet Fluorescence Microscopy. Neuroinformatics, 19(3), 433–446.[Link](https://doi.org/10.1007/s12021-020-09490-8)


It is also avivalbe from this [GitHub repo](https://github.com/Gubra-ApS/LSFM-mouse-brain-atlas/)



## Setup Instructions

### Using Conda

1. Install the Conda environment:
    `conda env create -f environment.yml`

2. Activate the environment:
    `conda activate delta_idisco`

### Using Pip

1. Install the necessary dependencies:
    `pip install -r requirements.txt`


## download requiered files

Use this googld drive [link](https://drive.google.com/drive/folders/1BzE3QRo38KOK5UYuipsq5TGubzbdSzap?usp=sharing) to download the atlas, annotation volume and paramater files 

## How to Use

Have a look at the `src\main.ipynb` notebook to see how to run a single brain or bsub a batch

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
