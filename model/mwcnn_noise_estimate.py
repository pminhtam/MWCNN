from .mwcnn_dgf import MWCNN_DGF
import torch.nn as nn
from guided_filter_pytorch.subnet import UNet
from guided_filter_pytorch.guided_filter import ConvGuidedFilter2
import torch
class MWCNN_noise(nn.Module):
    def __init__(self,n_colors=3):
        super(MWCNN_noise, self).__init__()
        self.unet = UNet(in_channels=n_colors+3, out_channels = 3)
        self.mwcnn_dgf = MWCNN_DGF(n_colors=n_colors)
        self.gf = ConvGuidedFilter2(radius=1)

    def forward(self, data,x_hr):

        pred_i, pred = self.mwcnn_dgf(data,x_hr)
        noise = x_hr - pred
        inp = torch.cat((x_hr,noise),dim=1)
        pred = self.unet(inp)
        return pred_i, pred