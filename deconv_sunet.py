r"""SUNet Deconvolution for Astronomical Images

This script uses an SUNet is to denoise images that were deconvolved using a Wiener filter.

Inspired by:

    - SUNet: Swin Transformer with UNet for Image Denoising (https://github.com/FanChiMao/SUNet)
      Authors: Chi-Mao Fan, Tsung-Jung Liu, Kuan-Hsien Liu

    - Tikhonet (https://github.com/CosmoStat/ShapeDeconv/blob/master/scripts/tikhonet/tikhonet_train.py)
      Authors: Fadi Nammour, François Lanusse, Hippolyte Karakostanoglou

"""

# Import dependencies
import torch
import torch.nn.functional as F
from collections import OrderedDict
import yaml
import sys
import numpy as np
import math
from skimage.transform import resize


# Utility functions

# Resize image while conserving flux
def resize_conserve_flux(img, size):
        orig_size = img.shape[0]
        img = resize(img, size, anti_aliasing=True)
        return img / (size[0]/orig_size)**2

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

# Deconvolution function
def deconv_sunet(noisy, 
                 psf, 
                 sampling_factor, 
                 SUNet_path,
                 model_dir,
                 model_name,
                 device='cpu'):
    
    r"""Applies SUNet deconvolution to image.
        This function takes a mutli-band image and its corresponding PSF (both in the direct space) and performs a deconvolution using a Wiener Filter.
        The Wiener filter is applied to the image in the Fourier space. The filtered image is then passed through SUNet to denoise it.

    Parameters
    ----------
    noisy   : 4D numpy array
        Noisy image in the direct space. Convention: (samples, channels, height, width).
    psf     : 3D numpy array
        PSF in the direct space. Convention: (channels, height, width).
    sampling_factor : scalar
        Factor by which the PSF is oversampled with respect to the noisy image.
    SUNet_path : string
        Path to SUNet repository.
    model_dir : string
        Path to SUNet model directory.
    model_name : string
        Name of the pre-trained SUNet model.
    device : string
        Device to run SUNet on.

    Returns
    -------
    res_sunet : 4D numpy array
        Deconvolved image in the direct space. 
        Output will be of size (samples, channels, height*sampling_factor, width*sampling_factor).
    tikho_deconv : 4D numpy array
        The intermediate deconvolved image in the direct space. 
        Output will be of size (samples, channels, height*sampling_factor, width*sampling_factor).
    """

    # Convert PSF to Fourier space
    rfft_psf = np.zeros((psf.shape[0], psf.shape[1], psf.shape[2]//2+1), dtype='complex64')
    for ch in range(psf.shape[0]):
        rfft_psf[ch] = np.fft.rfft2(np.fft.ifftshift(psf[ch]))

    # Tikhonov regularization weight (determined using line search)
    balance = 9e-3

    tikho_deconv = np.zeros((noisy.shape[0], noisy.shape[1], noisy.shape[2]*sampling_factor, noisy.shape[3]*sampling_factor))
    
    # Perform tikhonov deconvolution
    for i in range(noisy.shape[0]):
        for ch in range(noisy.shape[1]):
            tikho_deconv[i,ch], _ = wiener(resize_conserve_flux(noisy[i,ch], (tikho_deconv.shape[2], tikho_deconv.shape[3])), 
                                           rfft_psf[ch], 
                                           balance)


    # Import SUNet
    sys.path.insert(1, SUNet_path)
    from model.SUNet import SUNet_model

    # Normalize & scale tikhonov outputs (inputs to SUNet)
    # Done by subtracting mean and dividing by peak
    mean = np.mean(tikho_deconv, axis=(2,3), keepdims=True)
    peak = np.max(tikho_deconv, axis=(2,3), keepdims=True)
    tikho_deconv = tikho_deconv - mean 
    tikho_deconv /= peak

    # Convert to torch tensor
    input_ = torch.tensor(tikho_deconv).float()

    model_img = tikho_deconv.shape[2]
    stride = model_img//2

    def overlapped_square(timg, kernel=model_img, stride=stride):
        patch_images = []
        b, c, h, w = timg.size()
        X = int(math.ceil(max(h, w) / float(kernel)) * kernel)
        img = torch.zeros(1, 1, X, X).type_as(timg)  # 3, h, w
        mask = torch.zeros(1, 1, X, X).type_as(timg)

        img[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)] = timg
        mask[:, :, ((X - h) // 2):((X - h) // 2 + h), ((X - w) // 2):((X - w) // 2 + w)].fill_(1.0)

        patch = img.unfold(3, kernel, stride).unfold(2, kernel, stride)
        patch = patch.contiguous().view(b, c, -1, kernel, kernel)  # B, C, #patches, K, K
        patch = patch.permute(2, 0, 1, 4, 3)  # patches, B, C, K, K

        for each in range(len(patch)):
            patch_images.append(patch[each])

        return patch_images, mask, X
    
    # Load model
    with open(SUNet_path+'training.yaml', 'r') as config:
        opt = yaml.safe_load(config)

    def load_checkpoint(model, weights):
        checkpoint = torch.load(weights, map_location=torch.device(device))
        try:
            model.load_state_dict(checkpoint["state_dict"])
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] 
                new_state_dict[name] = v
            model.load_state_dict(new_state_dict)

    model = SUNet_model(opt)
    if device=='gpu':
        model.cuda()

    load_checkpoint(model, model_dir+model_name)
    model.eval()

    res_sunet = torch.zeros(input_.shape)

    for ch in range(input_.shape[1]):
        for ind in range(input_.shape[0]):
            with torch.no_grad():
                square_input_, mask, max_wh = overlapped_square(input_[ind, ch].unsqueeze(0).unsqueeze(0), kernel=model_img, stride=stride)
                output_patch = torch.zeros(square_input_[0].shape).type_as(square_input_[0])
                for i, data in enumerate(square_input_):
                    restored = model(square_input_[i])
                    if i == 0:
                        output_patch += restored
                    else:
                        output_patch = torch.cat([output_patch, restored], dim=0)

                B, C, PH, PW = output_patch.shape
                weight = torch.ones(B, C, PH, PH).type_as(output_patch)  # weight_mask

                patch = output_patch.contiguous().view(B, C, -1, model_img*model_img)
                patch = patch.permute(2, 1, 3, 0)  # B, C, K*K, #patches
                patch = patch.contiguous().view(1, C*model_img*model_img, -1)

                weight_mask = weight.contiguous().view(B, C, -1, model_img * model_img)
                weight_mask = weight_mask.permute(2, 1, 3, 0)  # B, C, K*K, #patches
                weight_mask = weight_mask.contiguous().view(1, C * model_img * model_img, -1)

                restored = F.fold(patch, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
                we_mk = F.fold(weight_mask, output_size=(max_wh, max_wh), kernel_size=model_img, stride=stride)
                restored /= we_mk

                res_sunet[ind,ch] = torch.masked_select(restored, mask.bool()).reshape(input_[ind,ch].unsqueeze(0).unsqueeze(0).shape)

    # Convert and un-normalize arrays
    res_sunet = np.squeeze(res_sunet.numpy()) * peak + mean
    tikho_deconv = tikho_deconv * peak + mean

    return res_sunet, tikho_deconv