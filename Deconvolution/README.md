# Ground-based Image Deconvolution with Swin Transformer UNet (SUNet)

We introduce a fast and efficient two-step deep learning-based deconvolution framework. The first step involves a Tikhonov deconvolution, followed by denoising with a Swin Transformer UNet (SUNet) architecture proposed by [Fan et al. (2022)](https://arxiv.org/pdf/2202.14009.pdf). 

This deconvolution method can be used to recover small-scale structures at high spatial resolution from ground-based multi-band observations. The algorithm is described in detail in [Akhaury et al. (2024)](https://doi.org/10.1051/0004-6361/202449495).   

<img src = "https://github.com/utsav-akhaury/SUNet/blob/main/Deconvolution/Data/deconv_sunet.png?raw=true" width="1000">  

## Installation

1) [Download and install Miniconda](http://conda.pydata.org/miniconda.html). Choose the Python 3.x version for your platform.

2) Open a Terminal (Linux/macOS) or Command Prompt (Windows) and run the following commands:
    ```bash
        conda update conda
        conda install git
        git clone https://github.com/utsav-akhaury/SUNet.git
        cd SUNet/Deconvolution
    ```
    
3) Create a conda environment and install all the required dependencies by running the following commands:

    ```bash
        conda env create -f conda_env.yml
    ```
 
4) Download the pre-trained PyTorch model from [Zenodo](https://doi.org/10.5281/zenodo.10287213).


## Code Overview

```bash
    SUNet/Deconvolution/
        Data/
            deconv_sunet.png
            hst.npy
            psf_vlt.npy
            vlt.npy
        README.rst
        conda_env.yml
        deconv_sunet.py
        tutorial.ipynb
```

* [Data](https://github.com/utsav-akhaury/SUNet/tree/main/Deconvolution/Data) is the directory containing the test images used in the tutorial notebook.
    * ``deconv_sunet.png`` is an image of a deconvolved galaxy.
    * ``Data/vlt.npy`` is a sample VLT image to be deconvolved.
    * ``Data/psf_vlt.npy`` is the PSF for the VLT image.
    * ``Data/hst.npy`` the corresponding HST image for comparison.
* [README.md](https://github.com/utsav-akhaury/SUNet/tree/main/Deconvolution/README.md) contains getting started information on installation and usage.
* [conda_env.yml](https://github.com/utsav-akhaury/SUNet/tree/main/Deconvolution/conda_env.yml) is a configuration file for Anaconda (Miniconda) that sets up a Python environment with all the required Python packages for using the SUNet Deconvolution code.
* [deconv_sunet.py](https://github.com/utsav-akhaury/SUNet/tree/main/Deconvolution/deconv_sunet.py) is the main script containing the deconvolution functions.
* [tutorial.ipynb](https://github.com/utsav-akhaury/SUNet/tree/main/Deconvolution/tutorial.ipynb) is a Jupyter notebook that guides you through the deconvolution process.


## Usage

1) Activate the ``sunet`` conda environment:

    ```bash
        conda activate sunet
    ```
2) Run the ``tutorial.ipynb`` notebook by replacing the input paths with the path to your cloned SUNet repository and the path to your SUNet model directory. The notebook will guide you through the deconvolution process. 
