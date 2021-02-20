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


    trans = transforms.ToPILImage()
    if not os.path.exists(args.save_img):
        os.makedirs(args.save_img)
    for mat_folder in mat_folders:
        save_mat_folder = os.path.join(args.save_img,mat_folder.split("/")[-1])
        for mat_file in glob.glob(os.path.join(mat_folder, '*')):
            mat_contents = sio.loadmat(mat_file)
            sub_image, y_gb, x_gb = mat_contents['image'], mat_contents['y_gb'][0][0], mat_contents['x_gb'][0][0]
            image_noise = transforms.ToTensor()(Image.fromarray(sub_image)).unsqueeze(0)
            image_noise_batch = image_noise.to(device)

            pred = model(image_noise_batch,0)
            pred = np.array(trans(pred[0].cpu()))
            if args.save_img != '':
                if not os.path.exists(save_mat_folder):
                    os.makedirs(save_mat_folder)
                # mat_contents['image'] = pred
                # print(mat_contents)
                print("save : ", os.path.join(save_mat_folder,mat_file.split("/")[-1]))
                data = {"image": pred, "y_gb": mat_contents['y_gb'][0][0], "x_gb": mat_contents['x_gb'][0][0],
                        "y_lc": mat_contents['y_lc'][0][0], "x_lc": mat_contents['x_lc'][0][0], 'size': mat_contents['size'][0][0],
                        "H": mat_contents['H'][0][0], "W": mat_contents['W'][0][0]}
                # print(data)
                sio.savemat(os.path.join(save_mat_folder,mat_file.split("/")[-1]), data)

if __name__ == "__main__":

    test_multi(args)

