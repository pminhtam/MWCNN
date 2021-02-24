import torch
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

def load_data(image_noise,burst_length):
    image_noise_hr = image_noise
    upscale_factor = int(math.sqrt(burst_length))
    image_noise = pixel_unshuffle(image_noise, upscale_factor)
    while len(image_noise) < burst_length:
        image_noise = torch.cat((image_noise,image_noise[-2:-1]),dim=0)
    if len(image_noise) > burst_length:
        image_noise = image_noise[0:8]
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop,image_noise_hr.unsqueeze(0)
def test(args):
    model = MWCNN_DGF(args)

    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:

    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = "model." + k  # remove `module.`
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

    all_noisy_imgs = scipy.io.loadmat(args.noise_dir)['BenchmarkNoisyBlocksSrgb']
    mat_re = np.zeros_like(all_noisy_imgs)
    i_imgs,i_blocks, _,_,_ = all_noisy_imgs.shape

    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(Image.fromarray(all_noisy_imgs[i_img][i_block]))
            image_noise, image_noise_hr = load_data(noise, args.burst_length)
            image_noise_hr = image_noise_hr.to(device)
            burst_noise = image_noise.to(device)
            begin = time.time()
            _, pred = model(burst_noise, image_noise_hr)
            pred = pred.detach().cpu()

            mat_re[i_img][i_block] = np.array(trans(pred[0]))
    return mat_re


if __name__ == "__main__":
    mat_re = test(args)
    mat = scipy.io.loadmat(args.noise_dir)
    del mat['BenchmarkNoisyBlocksSrgb']

    mat['DenoisedNoisyBlocksSrgb'] = mat_re
    # print(mat)
    scipy.io.savemat("SubmitSrgb.mat",mat)
