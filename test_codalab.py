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
from model.mwcnn import Model
from collections import OrderedDict
import glob
# from torchsummary import summary
import scipy.io as sio

torch.set_num_threads(4)
torch.manual_seed(0)
torch.manual_seed(0)

def test_multi(args):
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
    # except:
    #     print('=> no checkpoint file to be loaded.')    # model.load_state_dict(state_dict)
    #     exit(1)
    model.eval()
    model = model.to(device)
    trans = transforms.ToPILImage()
    torch.manual_seed(0)

    mat_folders = glob.glob(os.path.join(args.noise_dir, '*'))


    for mat_folder in mat_folders:
        for mat_file in glob.glob(os.path.join(mat_folder, '*')):
            mat_contents = sio.loadmat(mat_file)
            sub_image, y_gb, x_gb = mat_contents['image'], mat_contents['y_gb'][0][0], mat_contents['x_gb'][0][0]
            image_noise = transforms.ToTensor()(Image.fromarray(sub_image))
            image_noise_batch = image_noise.to(device)

            pred = model(image_noise_batch,0)
            pred = pred.detach().cpu()
            if args.save_img != '':
                if not os.path.exists(args.save_img):
                    os.makedirs(args.save_img)
                if not os.path.exists(mat_folder.replace(mat_folders,args.save_img)):
                    os.makedirs(mat_folder.replace(mat_folders,args.save_img))
                mat_contents['image'] = pred
                sio.savemat(mat_file.replace(mat_folder,args.save_img), mat_contents)


if __name__ == "__main__":

    test_multi(args)

