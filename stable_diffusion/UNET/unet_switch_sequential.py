from torch import nn
from torch.nn import functional as F
from UNET.unet_attentionblock import UNET_AttentionBlock
from UNET.unet_residualblock import UNET_ResidualBlock

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x


