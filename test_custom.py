import torch
import argparse
import utility
from model.mwcnn import Model
from torch.utils.data import DataLoader
# import h5py
from option import args
from data.data_provider import SingleLoader
from torchsummary import summary
from utils.metric import calculate_psnr,calculate_ssim
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
import math
from PIL import Image
import glob
import time
# from torchsummary import summary

torch.set_num_threads(4)
torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)
torch.manual_seed(0)

def test(args):
    model = Model(args)
    save_img = ''
    # summary(model,[[3,128,128],[0]])
    # exit()
    checkpoint_dir = "checkpoint/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_iter']
        state_dict = checkpoint['state_dict']
        model.model.load_state_dict(state_dict)
        print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    except:
        print('=> no checkpoint file to be loaded.')    # model.load_state_dict(state_dict)
        exit(1)
    model.eval()
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    for i in range(len(noisy_path)):
        noise = transforms.ToTensor()(Image.open(noisy_path[i]).convert('RGB')).unsqueeze(0)
        noise = noise.to(device)
        begin = time.time()
        # print(feedData.size())
        pred = model(noise,0)
        pred = pred.detach().cpu()
        gt = transforms.ToTensor()(Image.open(clean_path[i]).convert('RGB'))
        gt = gt.unsqueeze(0)
        psnr_t = calculate_psnr(pred, gt)
        ssim_t = calculate_ssim(pred, gt)
        print(i,"   UP   :  PSNR : ", str(psnr_t)," :  SSIM : ", str(ssim_t))
        if save_img != '':
            if not os.path.exists(args.save_img):
                os.makedirs(args.save_img)
            plt.figure(figsize=(15, 15))
            plt.imshow(np.array(trans(pred[0])))
            plt.title("denoise KPN DGF "+args.model_type, fontsize=25)
            image_name = noisy_path[i].split("/")[-1].split(".")[0]
            plt.axis("off")
            plt.suptitle(image_name+"   UP   :  PSNR : "+ str(psnr_t)+" :  SSIM : "+ str(ssim_t), fontsize=25)
            plt.savefig( os.path.join(args.save_img,image_name + "_" + args.checkpoint + '.png'),pad_inches=0)


if __name__ == "__main__":
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    test(args)

