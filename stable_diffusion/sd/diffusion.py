import torch
from torch import nn
from torch.nn import functional as F
from attention.cross_attention import CrossAttention
from attention.self_attention import SelfAttention
from UNET.unet_residualblock import UNET_ResidualBlock 
from UNET.unet_attentionblock import UNET_AttentionBlock
from time_embedding.time_embedding import TimeEmbedding
from UNET.unet_outputlayer import UNET_OutputLayer
from UNET.unet import UNET


class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_OutputLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return output