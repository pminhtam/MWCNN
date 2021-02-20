import os
import scipy.io as sio
import numpy as np
import glob
import cv2
from natsort import natsorted


def merge_image(save_dir, folder_dir):
  image_name = os.path.split(folder_dir)[-1]
  
  mat_file = os.path.join(folder_dir, '0.mat')
  mat_contents = sio.loadmat(mat_file)

  H, W = mat_contents['H'][0][0], mat_contents['W'][0][0]
  # print(H)
  image = np.zeros((H, W, 3), dtype=np.uint8)
  for mat_file in natsorted(glob.glob(os.path.join(folder_dir, '*'))):
    mat_contents = sio.loadmat(mat_file)
    sub_image, y_gb, x_gb = mat_contents['image'], mat_contents['y_gb'][0][0], mat_contents['x_gb'][0][0]
    y_lc, x_lc =  mat_contents['y_lc'][0][0], mat_contents['x_lc'][0][0]
    size = mat_contents['size'][0][0]

    image[y_gb:y_gb+size,x_gb:x_gb+size] = sub_image[y_lc:y_lc+size,x_lc:x_lc+size]

    
    

  cv2.imwrite(os.path.join(save_dir, '{}.png'.format(image_name)), image[...,::-1])

save_dir = '/root/codalab_re_img'
if not os.path.exists(save_dir):
  os.makedirs(save_dir)
# merge_image(save_dir, '/content/drive/MyDrive/CycleISP/CycleISP/MAI/255 (1)')
for folder_dir in natsorted(glob.glob('/root/codalab_re/*')):
  print(folder_dir)
  merge_image(save_dir, folder_dir)





