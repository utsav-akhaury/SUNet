from skimage.transform import resize
import numpy as np
import torch
import math


def overlapped_square(timg, kernel, stride):
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


def resize_batch_conserve_flux(images, target_shape, use_anti_aliasing=True):
    batch_size, channels, height, width = images.shape
    scale_factor = target_shape[0] / height
    
    resized_images = resize(
        images, 
        (batch_size, channels, *target_shape), 
        anti_aliasing=use_anti_aliasing
    )
    
    return resized_images / (scale_factor**2)

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

def compute_tikhonov_kernel(psf, balance):
    psf_centered = np.fft.ifftshift(psf, axes=(-2, -1))
    psf_fft = np.fft.rfft2(psf_centered, axes=(-2, -1))
    
    psf_conjugate = np.conj(psf_fft)
    psf_magnitude_squared = np.real(psf_fft * psf_conjugate)
    
    return psf_conjugate / (psf_magnitude_squared + balance)

def load_checkpoint(model, checkpoint_path, device="cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint["state_dict"])