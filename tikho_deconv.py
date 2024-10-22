r"""Deconvolution suing Tikhonov Regularization

Inspired by:

- Tikhonet (https://github.com/CosmoStat/ShapeDeconv/blob/master/scripts/tikhonet/tikhonet_train.py)
      Authors: Fadi Nammour, François Lanusse, Hippolyte Karakostanoglou

"""

import numpy as np

# Convert impulse response to transfer function
def ir2tf(imp_resp, shape):
    
    dim = 2
    # Zero padding and fill
    irpadded = np.zeros(shape)
    irpadded[tuple([slice(0, s) for s in imp_resp.shape])] = imp_resp
    # Roll for zero convention of the fft to avoid the phase problem. Work with odd and even size.
    for axis, axis_size in enumerate(imp_resp.shape):

        irpadded = np.roll(irpadded,
                        shift=-int(np.floor(axis_size / 2)),
                        axis=axis)

    return np.fft.rfftn(irpadded, axes=range(-dim, 0))

# Laplacian regularization
def laplacian_func(shape):
    
    impr = np.zeros([3,3])
    for dim in range(2):
        idx = tuple([slice(1, 2)] * dim +
                    [slice(None)] +
                    [slice(1, 2)] * (1 - dim))
        impr[idx] = np.array([-1.0,
                            0.0,
                            -1.0]).reshape([-1 if i == dim else 1
                                            for i in range(2)])
    impr[(slice(1, 2), ) * 2] = 4.0
    return ir2tf(impr, shape), impr

# Wiener filter
def wiener(image, psf, balance, laplacian=True):

    r"""Applies Wiener filter to image.
    This function takes an image in the direct space and its corresponding PSF in the
    Fourier space and performs a deconvolution using the Wiener Filter.

    Parameters
    ----------
    image   : 2D TensorFlow tensor
        Image in the direct space.
    psf     : 2D TensorFlow tensor
        PSF in the Fourier space (or K space).
    balance : scalar
        Weight applied to regularization.
    laplacian : boolean
        If true the Laplacian regularization is used else the identity regularization 
        is used.

    Returns
    -------
    tuple
        The first element is the filtered image in the Fourier space.
        The second element is the PSF in the Fourier space (also know as the Transfer
        Function).
    """

    trans_func = psf

    # Compute the regularization
    if laplacian:
        reg = laplacian_func(image.shape)[0]
        if psf.shape != reg.shape:
            trans_func = np.fft.rfft2(np.fft.ifftshift(psf).astype('float32'))  
        else:
            trans_func = psf
    
    arg1 = np.conj(trans_func).astype('complex64')
    arg2 = np.absolute(trans_func).astype('complex64') ** 2
    arg3 = balance
    if laplacian:
        arg3 *= np.absolute(laplacian_func(image.shape)[0]).astype('complex64')**2
    wiener_filter = arg1 / (arg2 + arg3)
    
    # Apply wiener in Fourier (or K) space
    wiener_applied = wiener_filter * np.fft.rfft2(image.astype('float32'))
    wiener_applied = np.fft.irfft2(wiener_applied)
    
    return wiener_applied, trans_func

# Apply Tikhonov Deconvolution
def apply_tikhonov_deconv(y_noisy, psf):

    """
    Apply Tikhonov Deconvolution to a set of images.

    Parameters
    ----------
    y_noisy : np.array
        Array of Noisy images. Convention is (N, H, W).
    psf : np.array
        Point Spread Function.

    Returns
    -------
    np.array
        Deconvolved images with Tikhonov regularization.
    """

    # Tikhonov Output
    y_tikho = np.zeros((y_noisy.shape), dtype='float32')

    rfft_psf = np.fft.rfft2(np.fft.ifftshift(psf))

    balance = 9e-3  # determined using line search
    for i in range(y_noisy.shape[0]):
        tikho_deconv_laplacian, _ = wiener(y_noisy[i], rfft_psf, balance)
        y_tikho[i] = tikho_deconv_laplacian

    return y_tikho