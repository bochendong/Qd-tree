import timm

import os
import torch
import torch.nn as nn
import torch.optim as optim

os.environ['HF_HOME'] = '/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree'
os.environ['TORCH_HOME'] = '/lustre/orion/bif146/world-shared/enzhi/qdt_imagenet/Qd-tree'

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size = 8, in_chans = 3, embed_dim = 768, num_patches = 196, bias = True):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = nn.Identity()
        self.num_patches = num_patches

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x
    
def get_model(model_type, num_classes, num_patches = 196, embed_dim = 768, to_size = (8, 8, 3)):
    if (model_type == 'vit_base_patch16_224'):
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        model.patch_embed = PatchEmbedding(patch_size = to_size[0], num_patches = num_patches, embed_dim = embed_dim)
        model.pos_embed = nn.Parameter(torch.randn(1, num_patches + model.num_prefix_tokens, embed_dim) * .02)

        return model
    else:
        print("Model is not supported")
    
