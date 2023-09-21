# Încărcare preconizată 2D UNET și modificați cu atenție temporală

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import einsum
import torch.utils.checkpoint
from einops import rearrange

import math

from diffusers import AutoencoderKL
from diffusers.models import UNet2DConditionModel

def get_unet(pretrained_model_name_or_path, revision, resolution=256, n_poses=5):
    # Încărcați straturi UNET preconjurate
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        revision="ebb811dd71cdc38a204ecbdd6ac5d580f529fd8c"
    )

    # Modify input layer to have 1 additional input channels (pose)
    weights = unet.conv_in.weight.clone()
    unet.conv_in = nn.Conv2d(4 + 2*n_poses, weights.shape[0], kernel_size=3, padding=(1, 1)) # Zgomot de intrare + n poze
    with torch.no_grad():
        unet.conv_in.weight[:, :4] = weights # greutăți originale
        unet.conv_in.weight[:, 4:] = torch.zeros(unet.conv_in.weight[:, 3:].shape) # Greutăți noi inițializate la zero

    return unet

'''
    This module takes in CLIP + VAE embeddings and outputs CLIP-compatible embeddings.
'''
# Definirea clasei modelului
class UNetDualPeopleEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_poses):
        super(UNetDualPeopleEncoder, self).__init__()

        # Aici puteți defini straturile rețelei UNet modificată
        # Exemplu:
        self.conv_in = nn.Conv2d(4 + 2 * n_poses, 64, kernel_size=3, padding=1)
        # Alte straturi pot fi adăugate aici

        # Definirea stratului de adaptare pentru încorporări
        self.embedding_adapter = EmbeddingAdapter(input_nc, output_nc)

    def forward(self, clip, vae):
        # Apelul straturilor modelului
        x = self.conv_in(torch.cat((clip, vae), dim=1))
        # Alte operații pot fi adăugate aici

        # Apelul stratului de adaptare pentru încorporări
        embeddings = self.embedding_adapter(x)

        return embeddings

# Definirea clasei EmbeddingAdapter
class EmbeddingAdapter(nn.Module):
    def __init__(self, input_nc=38, output_nc=4):
        super(EmbeddingAdapter, self).__init__()

        self.save_method_name = "adapter"

        self.pool =  nn.MaxPool2d(2)
        self.vae2clip = nn.Linear(1280, 768)

        self.linear1 = nn.Linear(54, 50) # 50 x 54 shape

        # Initialize weights
        with torch.no_grad():
            self.linear1.weight = nn.Parameter(torch.eye(50, 54))

    def forward(self, clip, vae):
        
        vae = self.pool(vae) # 1 4 80 64 --> 1 4 40 32
        vae = rearrange(vae, 'b c h w -> b c (h w)') # 1 4 20 16 --> 1 4 1280

        vae = self.vae2clip(vae) # 1 4 768

        # Concatenate
        concat = torch.cat((clip, vae), 1)

        # Encode
        concat = rearrange(concat, 'b c d -> b d c')
        concat = self.linear1(concat)
        concat = rearrange(concat, 'b d c -> b c d')

        return concat

# Exemplu de utilizare a modelului
if __name__ == "__main__":
    input_nc = 38  # Numărul de canale de intrare
    output_nc = 4  # Numărul de canale de ieșire
    n_poses = 5  # Numărul de poziții

    model = UNetDualPeopleEncoder(input_nc, output_nc, n_poses)
    print(model)  # Afișarea arhitecturii modelului