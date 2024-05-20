*******************************************************************
Ground-based Image Deconvolution with Swin Transformer UNet (SUNet)
*******************************************************************

We introduce a fast and efficient two-step deep learning-based deconvolution framework. The first step involves a Tikhonov deconvolution, followed by denoising with a Swin Transformer UNet (SUNet) architecture proposed by `Fan et al. (2022) <https://arxiv.org/pdf/2202.14009.pdf>`_. 

This deconvolution method can be used to recover small-scale structures at high spatial resolution from ground-based multi-band observations. The algorithm is described in detail in `Akhaury et al. <https://arxiv.org/pdf/2405.07842>`_.   

This is a fork of the `original implementation <https://github.com/FanChiMao/SUNet>`_ of the SUNet code in PyTorch by `Fan et al. (2022) <https://arxiv.org/pdf/2202.14009.pdf>`_. We thank the authors for making the code publicly available.

Installation
============

1) `Download and install Miniconda <http://conda.pydata.org/miniconda.html>`_. Choose the Python 3.x version for your platform.

2) Open a Terminal (Linux/macOS) or Command Prompt (Windows) and run the following commands:

    .. code-block:: bash

        conda update conda
        conda install git
        git clone https://github.com/utsav-akhaury/SUNet.git
        cd SUNet/Deconvolution

3) Create a conda environment and install all the required dependencies by running the following commands:

    .. code-block:: bash

        conda env create -f conda_env.yml
 
4) Download the pre-trained PyTorch model from `Zenodo <https://doi.org/10.5281/zenodo.10287213>`_.


Code Overview
=============

.. code-block:: bash

    SUNet/Deconvolution/
        Data/
            hst.npy
            psf_vlt.npy
            vlt.npy
        README.rst
        conda_env.yml
        deconv_sunet.py
        tutorial.ipynb

* ``Data`` is the directory containing the test images used in the tutorial notebook.
    * ``Data/vlt.npy`` is a sample VLT image to be deconvolved.
    * ``Data/psf_vlt.npy`` is the PSF for the VLT image.
    * ``Data/hst.npy`` the corresponding HST image for comparison.
* ``README.rst`` contains getting started information on installation and usage.
* ``conda_env.yml`` is a configuration file for Anaconda (Miniconda) that sets up a Python environment with all the required Python packages for using the SUNet Deconvolution code.
* ``decon_sunet.py`` is the main script containing the deconvolution functions.
* ``tutorial.ipynb`` is a Jupyter notebook that guides you through the deconvolution process.


Usage
=====

1) Activate the ``sunet`` conda environment:

    .. code-block:: bash

        conda activate sunet

2) Run the ``tutorial.ipynb`` notebook by replacing the input paths with the path to your cloned SUNet repository and the path to your SUNet model directory. The notebook will guide you through the deconvolution process. 