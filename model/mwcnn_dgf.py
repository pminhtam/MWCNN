from .mwcnn import MWCNN
import torch.nn as nn

from guided_filter_pytorch.guided_filter import ConvGuidedFilter2
class MWCNN_DGF(nn.Module):
    def __init__(self,n_resblocks=20,n_feats=64 ,n_colors=3):
        super(MWCNN_DGF, self).__init__()
        self.mwcnn = MWCNN(n_resblocks=n_resblocks,n_feats=n_feats ,n_colors=n_colors)
        self.gf = ConvGuidedFilter2(radius=1,n_colors=n_colors,n_bursts=4)

    def forward(self, data,x_hr):
        b, N, c, h, w = data.size()
        data = data.view(b*N,c,h,w)
        pred = self.mwcnn(data)

        b_hr, c_hr, h_hr, w_hr = x_hr.size()
        # print("x_hr  ",x_hr.size())
        x_hr_feed = x_hr.view(-1,c, h_hr, w_hr)
        data_feed = data.view(b,c*N,h,w)
        pred_feed = pred.view(b,c*N,h,w)
        # print(data_feed.size())
        # print(pred_feed.size())
        # print(x_hr_feed.size())

        out_hr = self.gf(data_feed, pred_feed, x_hr_feed)

        out_hr = out_hr.view(b,c, h_hr,w_hr)
        return data_feed.view(b,N,c,h,w),out_hr