import torch
import argparse
from model.mwcnn import Model

import numpy as np
import torchvision.transforms as transforms
from utils.training_util import load_checkpoint
from PIL import Image
import time
import scipy.io
from utils.raw_util import pack_raw,unpack_raw
from option import args
from collections import OrderedDict

def test(args):
    model = Model(args)
    checkpoint_dir = args.checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # try:
    checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
    start_epoch = checkpoint['epoch']
    global_step = checkpoint['global_iter']
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = "model." + k  # remove `module.`
        new_state_dict[name] = v
    model.load_state_dict(new_state_dict)
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
            noise = transforms.ToTensor()(pack_raw(all_noisy_imgs[i_img][i_block])).unsqueeze(0)
            noise = noise.to(device)
            begin = time.time()
            pred = model(noise,0)
            pred = np.array(pred.detach().cpu()[0]).transpose(1,2,0)
            pred = unpack_raw(pred)
            mat_re[i_img][i_block] = np.array(pred)

    return mat_re

if __name__ == "__main__":
    # argparse

    # args.noise_dir = '/home/dell/Downloads/FullTest/noisy'
    mat_re = test(args)

    mat = scipy.io.loadmat(args.noise_dir)
    # print(mat['BenchmarkNoisyBlocksSrgb'].shape)
    del mat['BenchmarkNoisyBlocksRaw']
    mat['DenoisedNoisyBlocksRaw'] = mat_re
    # print(mat)
    scipy.io.savemat("SubmitRaw.mat",mat)
