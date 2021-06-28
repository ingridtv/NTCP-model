# Dose statistics and NTCP calculation from DICOM data


### Purpose

The program contains functionality to perform dose calculations and generate
dose statistics from DICOM (.dcm) data. The statistics can be correlated with
patient outcome through various analyses.



## About

The program can calculate dose/DVH statistics, fit the Lyman-Kutcher-Burman 
(LKB) model for NTCP, and perform a voxel-by-voxel comparison of dose distributions. 
To do this, the program uses RTSS, RTDOSE images in particular.

The functionality was developed using images from a group of prostate cancer
patients, with the adaptations that are suitable for that patient group. In
addition, outcome was assessed using patient-reported outcomes (PROs), but the
program could be easily adapted to utilize other types of outcome measures.



## Requirements

The program was developed using Python 3.8. Compatibility with earlier Python
versions cannot be guaranteed.

The following python packages and versions are required/suggested:
* numpy >= 1.20.1
* matplotlib => 3.3.4
* scipy >= 1.6.1
* pandas >= 1.2.2
* scikit-learn >= 0.24.0
* pydicom >= 2.1.2
* dicompyler-core >= 0.5.5
* openpyxl >= 3.0.6
* pathlib >=  1.0.1

To perform the voxel-based analysis, the SimpleElastix package (<https://github.com/SuperElastix/SimpleElastix>) needs to be compiled and available.


#### Installation

To install the required packages, open the project folder in the command line
or  terminal and run the following command:

```terminal
pip install -r requirements.txt
```

The SimpleElastix software needs compilation and should be installed separately.

