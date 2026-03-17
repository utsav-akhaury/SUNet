import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from thop import profile

from model.helpers.layers_and_patches import BasicLayer, BasicLayer_up, PatchEmbed, PatchMerging, UpSample


class SUNet(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3

        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, out_chans=3,
                 embed_dim=96, depths=[2, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample=None, **kwargs):  # <=== #
        super(SUNet, self).__init__()

        self.out_chans = out_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample
        self.prelu = nn.PReLU()
        self.conv_first = nn.Conv2d(in_chans, embed_dim, 3, 1, 1)
        self.patch_size = patch_size  # <=== #
        self.image_size = img_size  # <=== #
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        def _block_drop_path(i_layer):
            start = sum(depths[:i_layer])
            return dpr[start:start + depths[i_layer]]

        def _layer_resolution(layer_idx):
            return (patches_resolution[0] // (2 ** layer_idx),
                    patches_resolution[1] // (2 ** layer_idx))

        def _make_down_layer(i_layer):
            return BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                              input_resolution=_layer_resolution(i_layer),
                              depth=depths[i_layer],
                              num_heads=num_heads[i_layer],
                              window_size=window_size,
                              mlp_ratio=self.mlp_ratio,
                              qkv_bias=qkv_bias, qk_scale=qk_scale,
                              drop=drop_rate, attn_drop=attn_drop_rate,
                              drop_path=_block_drop_path(i_layer),
                              norm_layer=norm_layer,
                              downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                              use_checkpoint=use_checkpoint)

        def _make_concat_linear(i_layer):
            if i_layer == 0:
                return nn.Identity()
            base_dim = int(embed_dim * 2 ** (self.num_layers - 1 - i_layer))
            return nn.Linear(2 * base_dim, base_dim)

        def _make_up_layer(i_layer):
            idx = self.num_layers - 1 - i_layer
            input_res = _layer_resolution(idx)
            dim = int(embed_dim * 2 ** idx)
            if i_layer == 0:
                return UpSample(input_resolution=input_res,
                                in_channels=dim,
                                scale_factor=2)
            return BasicLayer_up(dim=dim,
                                 input_resolution=input_res,
                                 depth=depths[idx],
                                 num_heads=num_heads[idx],
                                 window_size=window_size,
                                 mlp_ratio=self.mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=_block_drop_path(idx),
                                 norm_layer=norm_layer,
                                 upsample=UpSample if (i_layer < self.num_layers - 1) else None,
                                 use_checkpoint=use_checkpoint)

        self.layers = nn.ModuleList(_make_down_layer(i_layer) for i_layer in range(self.num_layers))
        self.layers_up = nn.ModuleList(_make_up_layer(i_layer) for i_layer in range(self.num_layers))
        self.concat_back_dim = nn.ModuleList(_make_concat_linear(i_layer)
                                              for i_layer in range(self.num_layers))

        self.norm = norm_layer(self.num_features)
        self.norm_up = norm_layer(self.embed_dim)

        if self.final_upsample == "bilinear":
            self.up = nn.Upsample(scale_factor=patch_size, mode=final_upsample, align_corners=False)
            self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.out_chans, kernel_size=3, stride=1,
                                    padding=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}


    def forward_features(self, x):
        residual = x
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)

        return x, residual, x_downsample

    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x, x_downsample[3 - inx]], -1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)

        return x

    def patch_unembedded(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "input features has wrong size"

        if self.final_upsample == 'bilinear':
            factor = self.patch_size
            x = x.view(B, H, W, -1)
            x = x.permute(0, 3, 1, 2)
            x = self.up(x)

        return x

    def forward(self, x):
        x = self.conv_first(x)
        x, residual, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.patch_unembedded(x)
        out = self.output(x)
        return out

    def flops(self):
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        flops += self.num_features * self.out_chans
        return flops


if __name__ == '__main__':
    from model.utils import network_parameters

    height = 128
    width = 128
    x = torch.randn((1, 3, height, width))  # .cuda()
    model = SUNet(img_size=128, patch_size=1, in_chans=3, out_chans=3,
                  embed_dim=32, depths=[6, 6, 6, 6],
                  num_heads=[8, 8, 8, 8],
                  window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=2,
                  drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                  norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                  use_checkpoint=False, final_upsample="bilinear")  # <=== # # .cuda()
    print('input image size: (%d, %d)' % (height, width))
    print('FLOPs: %.4f G' % (model.flops() / 1e9))
    print('model parameters: ', network_parameters(model))
    print('output image size: ', x.shape)
    flops, params = profile(model, (x,))
    print(flops)
    print(params)
