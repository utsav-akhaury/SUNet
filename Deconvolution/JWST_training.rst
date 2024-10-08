*******************************************************************
Re-training SUNet for JWST data
*******************************************************************

Requirements for training data
==============================

For each object, we would require:

1) A noisy ground-based image

2) The corresponding ground-based PSF

3) A target space-based image (JWST)

We would need atleast ``~25,000`` objects for the training set.


Specifications
==============

1) The noisy image and the target image should have dimensions that are multiples of each other and powers of ``2``. 
    * For example, for our previously trained SUNet, the VLT images were ``32 × 32`` pixels, while the HST images were ``128 × 128`` pixels. This was consistent with the fact that the pixel scales also differ by a factor of ``4``.

2) The noisy and the target images can be left at their oringal dynamic range.
    * SUNet will by default pre-process the images by normalizing them before training.

3) The PSF should be at the pixel scale of the desired target output.
    * In our previous case, we wanted all images to have a resolution equal to that of the HST, so our PSF was also ``128 × 128`` pixels.

4) It would be interesting to have PSFs with FWHMs (Full Width at Half Maximum) ranging from ``0.5″`` to ``0.9″`` depending on the region of the sky, to make our training more robust.
    * Previously, our PSF had an FWHM of ``~0.75″`` for the entire training sample.
    
5) For each object, a corresponding normalized PSF is required, with a flux of ``1``.
    * i.e. the sum of all the pixel values should be ``1``.
    
6) To filter objects of interest from the HST catalogue, we previously chose the following criteria (the last 2 conditions ensure that we exclude stars or point-sources in our dataset):
    * ``MAG_AUTO < 26`` (AB magnitude in SExtractor “AUTO” aperture)
    * ``Flux_Radius_80 > 10`` (80% enclosed flux radius in pixels)
    * ``FWHM > 10`` (full width at half maximum in pixels)