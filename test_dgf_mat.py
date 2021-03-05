import torch
from utils.metric import calculate_psnr,calculate_ssim
import os
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
from PIL import Image
import time
import scipy.io
from option import args
from model.mwcnn_dgf import MWCNN_DGF
from collections import OrderedDict
from data.data_provider import pixel_unshuffle
import math
# from torchsummary import summary

torch.manual_seed(0)
def load_data_split(image_noise,burst_length):
    image_noise_hr = image_noise
    upscale_factor = int(math.sqrt(burst_length))
    image_noise = pixel_unshuffle(image_noise, upscale_factor)
    while len(image_noise) < burst_length:
        image_noise = torch.cat((image_noise,image_noise[-2:-1]),dim=0)
    if len(image_noise) > burst_length:
        image_noise = image_noise[0:8]
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop,image_noise_hr.unsqueeze(0)
from data.dataset_utils import burst_image_filter
def load_data_filter(image_noise, burst_length):
    image_noise_hr = image_noise
    image_noise = burst_image_filter(image_noise_hr)
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop, image_noise_hr.unsqueeze(0)

def test(args):
    model = MWCNN_DGF(args)
    if args.data_type == 'rgb':
        load_data = load_data_split
    elif args.data_type == 'filter':
        load_data = load_data_filter
    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:

    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = "model." + k[6:]  # remove `module.`
    #     new_state_dict[name] = v
    model.load_state_dict(state_dict)
    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    # except:
    #     print('=> no checkpoint file to be loaded.')    # model.load_state_dict(state_dict)
    #     exit(1)
    model.eval()
    model = model.to(device)
    trans = transforms.ToPILImage()
    torch.manual_seed(0)

    all_noisy_imgs = scipy.io.loadmat(args.noise_dir)['ValidationNoisyBlocksSrgb']
    all_clean_imgs = scipy.io.loadmat(args.gt_dir)['ValidationGtBlocksSrgb']
    # noisy_path = sorted(glob.glob(args.noise_dir+ "/*.png"))
    # clean_path = [ i.replace("noisy","clean") for i in noisy_path]
    i_imgs,i_blocks, _,_,_ = all_noisy_imgs.shape
    psnrs = []
    ssims = []
    # print(noisy_path)
    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block]))
            image_noise, image_noise_hr = load_data(noise, args.burst_length)
            image_noise_hr = image_noise_hr.to(device)
            burst_noise = image_noise.to(device)
            begin = time.time()
            pred_i , pred = model(burst_noise, image_noise_hr)
            pred = pred.detach().cpu()
            gt = transforms.ToTensor()((Image.fromarray(all_clean_imgs[i_img][i_block])))
            gt_lr, gt = load_data(gt, args.burst_length)
            psnr_t = calculate_psnr(pred, gt)
            ssim_t = calculate_ssim(pred, gt)
            psnrs.append(psnr_t)
            ssims.append(ssim_t)
            print(i_img, "      :  PSNR : ", str(psnr_t), " :  SSIM : ", str(ssim_t))
            psnr_i1 = calculate_psnr(pred_i[-1], gt_lr[-1])
            ssim_i1 = calculate_ssim(pred_i[-1], gt_lr[-1])
            print("   Image 4   :  PSNR : ", str(psnr_i1), " :  SSIM : ", str(ssim_i1))
            psnr_i2 = calculate_psnr(np.mean(pred_i,axis=0), np.mean(gt_lr[-1]))
            ssim_i2 = calculate_ssim(np.mean(pred_i,axis=0), np.mean(gt_lr[-1]))
            print("   Image Mean   :  PSNR : ", str(psnr_i2), " :  SSIM : ", str(ssim_i2))
            if args.save_img != '':
                if not os.path.exists(args.save_img):
                    os.makedirs(args.save_img)
                plt.figure(figsize=(15, 15))
                plt.imshow(np.array(trans(pred[0])))
                plt.title("denoise KPN DGF " + args.model_type, fontsize=25)
                image_name = str(i_img)
                plt.axis("off")
                plt.suptitle(image_name + "   UP   :  PSNR : " + str(psnr_t) + " :  SSIM : " + str(ssim_t), fontsize=25)
                plt.savefig(os.path.join(args.save_img, image_name + "_" + args.checkpoint + '.png'), pad_inches=0)
    print("   AVG   :  PSNR : "+ str(np.mean(psnrs))+" :  SSIM : "+ str(np.mean(ssims)))


if __name__ == "__main__":
    test(args)

