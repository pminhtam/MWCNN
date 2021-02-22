from .mwcnn_dgf import MWCNN_DGF
import torch.nn as nn
from guided_filter_pytorch.subnet import UNet
from guided_filter_pytorch.guided_filter import ConvGuidedFilter2
import torch
class MWCNN_noise(nn.Module):
    def __init__(self):
        super(MWCNN_noise, self).__init__()
        self.unet = UNet(in_channels=6, out_channels = 3)
        self.mwcnn_dgf = MWCNN_DGF(n_colors=3)
        self.gf = ConvGuidedFilter2(radius=1)

    def forward(self, data,x_hr):

        pred_i, pred = self.mwcnn_dgf(data,x_hr)
        noise = x_hr - pred
        inp = torch.cat((x_hr,noise),dim=1)
        pred = self.unet(inp)
        return pred_i, pred