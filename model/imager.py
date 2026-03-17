
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np

from model.SUNet import SUNet
from model.utils import get_device
class SUNetImager(nn.Module):

    def __init__(
        self,
        checkpoint_path: str | Path = None,
        device: str = "auto",
        img_size=224, patch_size=4, in_chans=3, out_chans=3,
        embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
        window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
        norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
        use_checkpoint=False, final_upsample=None, **kwargs
    ):
        super(SUNetImager, self).__init__()

        self.sunet = SUNet(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, out_chans=out_chans,
            embed_dim=embed_dim, depths=depths, num_heads=num_heads,
            window_size=window_size, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate,
            norm_layer=norm_layer, ape=ape, patch_norm=patch_norm,
            use_checkpoint=use_checkpoint, final_upsample=final_upsample, **kwargs)
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.patch_size = img_size
        self.stride_step = patch_size

        if device == "auto":
            self.device = get_device()
            print(f"Using device: {self.device}")
        self.to(self.device)

        if self.checkpoint_path is not None:
            model_state_dict = torch.load(
                self.checkpoint_path, map_location=self.device
            )["model_state_dict"]
            self.sunet.load_state_dict(model_state_dict)
            print(f"Loaded weights from {self.checkpoint_path}")

        self.sunet.eval()

    def _normalize_image(self, image):
        image = torch.as_tensor(image, dtype=torch.float32, device=self.device)

        mean = torch.mean(image, dim=(2, 3), keepdim=True)
        
        peak = torch.amax(image, dim=(2, 3), keepdim=True)
        
        normalized_image = (image - mean) / peak
        return normalized_image, mean, peak
    
    def _denoise_sunet(self, image):
        """
        Deconvolves an image using the SUNet model by processing it in overlapping patches.
        """
        from Deconvolution.deconv_sunet import normalize_patch_overlaps, prepare_patches_for_folding, prepare_patches_for_inference
        import torch.nn.functional as F
        image_tensor, mean_value, peak_value = self._normalize_image(image)
        
        batch_count, channel_count, height, width = image_tensor.shape
        target_resolution = (height, width)

        
        patches = F.unfold(
            image_tensor.to(self.device), 
            kernel_size=self.patch_size, 
            stride=self.stride_step
        )
        num_patches = patches.shape[-1]
        
        patches = prepare_patches_for_inference(patches, batch_count, channel_count, self.patch_size)
        
        with torch.no_grad():
            restored_patches = self.sunet(patches) 
            
        restored_patches = prepare_patches_for_folding(restored_patches, batch_count, num_patches, channel_count, self.patch_size)

        restored_image = F.fold(
            restored_patches, 
            output_size=target_resolution, 
            kernel_size=self.patch_size, 
            stride=self.stride_step
        )
        
        normalized_output = normalize_patch_overlaps(restored_image,
                                                    num_patches,
                                                    channel_count,
                                                    self.patch_size,
                                                    self.stride_step,
                                                    target_resolution)
        return normalized_output * peak_value + mean_value


    def forward(self, image: torch.tensor,
        psf: torch.tensor)-> torch.tensor:
        from Deconvolution.deconv_sunet import tikhonov_filter
        balance = 9e-3
        tikho_deconv = tikhonov_filter(image, psf, balance)
        print(f"Tikhonov deconvolution completed. Shape: {tikho_deconv.shape}")
        x = self._denoise_sunet(tikho_deconv)
        print(f"SUNet deconvolution completed. Shape: {x.shape}")
        return x