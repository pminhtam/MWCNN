import torch
import argparse
import utility
from model.mwcnn import Model
from torch.utils.data import DataLoader
# import h5py
from option import args
from data.data_provider import SingleLoader
torch.set_num_threads(4)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
# import model
from torchsummary import summary
from utils.metric import calculate_psnr
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from collections import OrderedDict

if __name__ == "__main__":


    data_set = SingleLoader(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=1,
        shuffle=False,
        num_workers=4
    )
    model = Model(args)

    state_dict = torch.load('experiment/checkpoint')
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = "model."+ k  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
    # model.load_state_dict(state_dict)
    model.eval()
    trans = transforms.ToPILImage()
    for epoch in range(10):
        for step, (noise, gt) in enumerate(data_loader):
            pred = model(noise,[10])
            print(pred.size())
            plt.subplot(1, 2, 1)
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise ", fontsize=20)
            plt.subplot(1,2,2)
            plt.imshow(np.array(trans(gt[0])))
            plt.show()
            print("PSNR  : ",calculate_psnr(pred,gt))    # print(model)
    # print(summary(model,[(3,512,512),[8]]))

