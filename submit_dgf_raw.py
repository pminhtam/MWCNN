import torch
import argparse
from model.mwcnn_dgf import MWCNN_DGF
from model.mwcnn_noise_estimate import MWCNN_noise
import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
from PIL import Image
import time
import scipy.io
from utils.raw_util import pack_raw,unpack_raw
import math
from data.data_provider import pixel_unshuffle
from option import args

def load_data(image_noise, burst_length):
    image_noise_hr = image_noise
    upscale_factor = int(math.sqrt(burst_length))
    image_noise = pixel_unshuffle(image_noise, upscale_factor)
    while len(image_noise) < burst_length:
        image_noise = torch.cat((image_noise, image_noise[-2:-1]), dim=0)
    if len(image_noise) > burst_length:
        image_noise = image_noise[0:8]
    image_noise_burst_crop = image_noise.unsqueeze(0)
    return image_noise_burst_crop, image_noise_hr.unsqueeze(0)
def test(args):
    if  args.model_type == "DGF":
        model = MWCNN_DGF(n_colors=args.n_colors)
    elif  args.model_type == "noise":
        model = MWCNN_noise(n_colors=args.n_colors)
    else:
        print(" Model type not valid")
        return
    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    model.eval()
    model = model.to(device)
    trans = transforms.ToPILImage()
    torch.manual_seed(0)
    all_noisy_imgs = scipy.io.loadmat(args.noise_dir)['BenchmarkNoisyBlocksRaw']
    mat_re = np.zeros_like(all_noisy_imgs)
    i_imgs,i_blocks, _,_ = all_noisy_imgs.shape

    for i_img in range(i_imgs):
        for i_block in range(i_blocks):
            noise = transforms.ToTensor()(pack_raw(all_noisy_imgs[i_img][i_block]))
            image_noise, image_noise_hr = load_data(noise, args.burst_length)
            image_noise_hr = image_noise_hr.to(device)
            burst_noise = image_noise.to(device)
            begin = time.time()
            _, pred = model(burst_noise,image_noise_hr)
            pred = np.array(pred.detach().cpu()[0]).transpose(1,2,0)
            pred = unpack_raw(pred)
            mat_re[i_img][i_block] = np.array(pred)

    return mat_re

if __name__ == "__main__":
    # argparse
    #
    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    mat_re = test(args)

    mat = scipy.io.loadmat(args.noise_dir)
    # print(mat['BenchmarkNoisyBlocksSrgb'].shape)
    del mat['BenchmarkNoisyBlocksRaw']
    mat['DenoisedNoisyBlocksRaw'] = mat_re
    # print(mat)
    scipy.io.savemat("SubmitRaw.mat",mat)
