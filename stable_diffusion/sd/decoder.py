import torch
from torch import nn 
from torch.nn import functional as F
from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int): 
        super().__init__()
        self.group_norm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residue = x 
        n, c, h, w = x.shape
        x = x.view(n, c, h * w) # [B, F, H * W]
        x = x.transpose(-1, -2) # [B, H * W, F]
        x = self.attention(x) # [B, H * W, F]
        x = x.transpose(-1, -2) # [B, F, H * W]
        x = x.view((n, c, h, w))
        x += residue
        return x


    

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
            super().__init__()
            self.group_norm_1 = nn.GroupNorm(32, in_channels)
            self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
            self.group_norm_2 = nn.GroupNorm(32, out_channels)
            self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

            if in_channels == out_channels:
                self.residual_layer = nn.Identity()
            else:
                self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
            Input: x [B, C, H, W]

        """
        residue = x 
        x = self.group_norm_1(x)
        x = F.silu(x)
        x = self.conv_1(x)
        x = self.group_norm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)
        return x + residue 

class VAE_Decoder(nn.Sequential):
    def __init__(self):
        super.__init__(
            nn.Conv2d(4, 4, kernel_size=1, padding=0), # [B, 4, H / 8, W / 8] -> [B, 4, H / 8, W / 8]
            nn.Conv2d(4, 512, kernel_size=3, padding=1), # [B, 4, H / 8, W / 8] -> [B, 512, H / 8, W / 8]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 8, W / 8]
            VAE_AttentionBlock(512), # [B, 512, H / 8, W / 8]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 8, W / 8]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 8, W / 8]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 8, W / 8]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 8, W / 8]
            
            # Repeats the rows and columns of the data by scale_factor (like when you resize an image by doubling its size).
            nn.Upsample(scale_factor=2), # [B, 512, H / 4, W / 4]
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # [B, 512, H / 4, W / 4]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 4, W / 4]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 4, W / 4]
            VAE_ResidualBlock(512, 512), # [B, 512, H / 4, W / 4]
            
            nn.Upsample(scale_factor=2), # [B, 512, H / 2, W / 2]
            nn.Conv2d(512, 512, kernel_size=3, padding=1), # [B, 512, H / 2, W / 2]
            VAE_ResidualBlock(512, 256), # [B, 256, H / 2, W / 2]
            VAE_ResidualBlock(256, 256), # [B, 256, H / 2, W / 2]
            VAE_ResidualBlock(256, 256), # [B, 256, H / 2, W / 2]
        
            nn.Upsample(scale_factor=2), # [B, 256, H, W]
            nn.Conv2d(256, 256, kernel_size=3, padding=1), # [B, 256, H, W]
            VAE_ResidualBlock(256, 128), # [B, 128, H, W]
            VAE_ResidualBlock(128, 128), # [B, 128, H, W]
            VAE_ResidualBlock(128, 128), # [B, 128, H, W]
            
            nn.GroupNorm(32, 128), # [B, 128, H, W]
            nn.SiLU(), # [B, 128, H, W]
         
            nn.Conv2d(128, 3, kernel_size=3, padding=1), # [B, 3, H, W]
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
            Input: x [B, 4, H/8, W/8]
            Output: x [B, 3, H, W]
        '''
      
        # Remove the scaling added by the Encoder.
        x /= 0.18215

        for module in self:
            x = module(x)

        return x