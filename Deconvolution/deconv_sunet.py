r"""Ground-based Image Deconvolution with Swin Transformer UNet (SUNet)

This script uses an SUNet to denoise images that were deconvolved using Tikhonov regularisation.

Inspired by:

    - SUNet: Swin Transformer with UNet for Image Denoising (https://github.com/FanChiMao/SUNet)
      Authors: Chi-Mao Fan, Tsung-Jung Liu, Kuan-Hsien Liu

    - Tikhonet (https://github.com/CosmoStat/ShapeDeconv/blob/master/scripts/tikhonet/tikhonet_train.py)
      Authors: Fadi Nammour, François Lanusse, Hippolyte Karakostanoglou

"""

# Import dependencies
from pathlib import Path
import torch
import torch.nn.functional as F
import yaml
import sys
import numpy as np

# Import SUNet
SUNET_PATH = '/home/fnammour/code/SUNet/'
sys.path.insert(1, SUNET_PATH)
from model.SUNet import SUNet_model

from Deconvolution.deconv_utils import compute_tikhonov_kernel, resize_batch_conserve_flux, overlapped_square, load_checkpoint

def tikhonov_filter(noisy_stack, psf_stack, balance=9e-3):
    psf_shape = psf_stack.shape
    psf_height, psf_width = psf_shape[-2], psf_shape[-1]
    
    target_shape = (psf_height, psf_width)
    
    kernel = compute_tikhonov_kernel(psf_stack, balance)
    
    upsampled_images = resize_batch_conserve_flux(noisy_stack, target_shape)
    image_fft = np.fft.rfft2(upsampled_images, axes=(-2, -1))

    deconvolved_fft = image_fft * kernel[np.newaxis, ...]
    
    return np.fft.irfft2(
        deconvolved_fft, 
        s=target_shape, 
        axes=(-2, -1)
    )

def normalize_image(image, device=torch.device("cpu")):
    mean = np.mean(image, axis=(2,3), keepdims=True)
    peak = np.max(image, axis=(2,3), keepdims=True)
    normalized_image = image - mean 
    normalized_image /= peak
    normalized_image = torch.from_numpy(normalized_image).float().to(device)
    return normalized_image, mean, peak

def instantiate_and_load_sunet(checkpoint_path:str, device=torch.device("cpu")):
    with open(Path(SUNET_PATH) / 'training.yaml', 'r') as config:
        config = yaml.safe_load(config)
    model = SUNet_model(config).to(device)
    load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    return model

def prepare_patches_for_inference(unfolded_patches, batch_count, channel_count, patch_size):
    """Reshapes unfolded patches into a batch of 2D images for the model."""
    num_patches = unfolded_patches.shape[-1]
    return (unfolded_patches
            .view(batch_count, channel_count, patch_size, patch_size, num_patches)
            .permute(0, 4, 1, 2, 3)
            .reshape(-1, channel_count, patch_size, patch_size))

def prepare_patches_for_folding(output_patches, batch_count, num_patches, channel_count, patch_size):
    """Reshapes model output patches back into the column format required by F.fold."""
    return (output_patches
            .view(batch_count, num_patches, channel_count, patch_size, patch_size)
            .permute(0, 2, 3, 4, 1)
            .reshape(batch_count, channel_count * patch_size**2, num_patches))

def normalize_patch_overlaps(folded_image_sum, num_patches, channel_count, patch_size, stride, target_resolution):
    """
    Normalizes a folded image by dividing by the number of overlaps per pixel.
    
    This prevents 'bright grids' or artifacts caused by F.fold summing values 
    in overlapping regions.
    """
    device = folded_image_sum.device
    batch_count = folded_image_sum.shape[0]
    
    patch_weight_ones = torch.ones(
        (batch_count, channel_count * patch_size**2, num_patches), 
        device=device
    )

    overlap_counts = F.fold(
        patch_weight_ones, 
        output_size=target_resolution, 
        kernel_size=patch_size, 
        stride=stride
    )

    return folded_image_sum / overlap_counts


def deconv_sunet(image, model, device=torch.device("cpu")):
    """
    Deconvolves an image using the SUNet model by processing it in overlapping patches.
    """
    image_tensor, mean_value, peak_value = normalize_image(image=image, device=device)
    
    config_path = Path(SUNET_PATH) / 'training.yaml'
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    patch_size = config["SWINUNET"]["IMG_SIZE"]
    stride_step = config["TRAINING"]["TRAIN_PS"]
    
    batch_count, channel_count, height, width = image_tensor.shape
    target_resolution = (height, width)

    
    patches = F.unfold(
        image_tensor.to(device), 
        kernel_size=patch_size, 
        stride=stride_step
    )
    num_patches = patches.shape[-1]
    
    patches = prepare_patches_for_inference(patches, batch_count, channel_count, patch_size)
    
    with torch.no_grad():
        restored_patches = model(patches) 
        
    restored_patches = prepare_patches_for_folding(restored_patches, batch_count, num_patches, channel_count, patch_size)

    restored_image = F.fold(
        restored_patches, 
        output_size=target_resolution, 
        kernel_size=patch_size, 
        stride=stride_step
    )
    
    normalized_output = normalize_patch_overlaps(restored_image,
                                                 num_patches,
                                                 channel_count,
                                                 patch_size,
                                                 stride_step,
                                                 target_resolution)
    final_image = normalized_output.cpu().detach().numpy()
    return np.squeeze(final_image) * peak_value + mean_value