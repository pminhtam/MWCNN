import cv2
import os
import scipy.io as sio
import shutil
import glob
from tqdm import tqdm

patch_size = 256
offset = 16

size = size = patch_size - 2 * offset

def crop_image(image_path, parrent_dir):
  image_name = os.path.splitext(os.path.split(image_path)[-1])[0]
  folder_dir = os.path.join(parrent_dir, image_name)
  
  # Remove folder
  if os.path.exists(folder_dir):
    shutil.rmtree(folder_dir)
  os.mkdir(folder_dir)

  image = cv2.imread(image_path)[...,::-1]
  index = 0
  H, W = image.shape[:2]
  # print((H, W))

  # Crop
  for ih in range(0, H // size * size , size):
    for iw in range(0, W // size * size , size):
      file_mat = os.path.join(folder_dir, '{}.mat'.format(index))
      if ih == 0 and iw == 0:
        crop_image = image[ih:ih+patch_size, iw:iw+patch_size]
        x_lc = y_lc = 0
      elif ih == 0:
        crop_image = image[ih:ih+patch_size, iw-offset:iw+patch_size-offset]
        y_lc = 0 ; x_lc = offset
      elif iw == 0:
        crop_image = image[ih-offset:ih+patch_size-offset, iw:iw+patch_size]
        y_lc = offset ; x_lc = 0
      else:
        crop_image = image[ih-offset:ih-offset+patch_size, iw-offset:iw-offset+patch_size]
        y_lc = x_lc = offset

      assert crop_image.shape[:2] == (patch_size, patch_size)
      data = {"image": crop_image, "y_gb":ih, "x_gb": iw, "y_lc": y_lc, "x_lc": x_lc, 'size': size, "H": H, "W": W}
      sio.savemat(file_mat, data)
      index += 1

  x_last = W - size
  for ih in range(0, H // size * size , size):
    file_mat = os.path.join(folder_dir, '{}.mat'.format(index))
    if ih == 0:
      crop_image = image[ih:ih+patch_size, x_last-offset*2:x_last-offset*2+patch_size]
      y_lc = 0 ; x_lc = 2*offset
    else:
      crop_image = image[ih-offset:ih-offset+patch_size, x_last-2*offset:x_last-2*offset+patch_size]
      y_lc = offset ; x_lc = 2*offset

    assert crop_image.shape[:2] == (patch_size, patch_size)
    data = {"image": crop_image, "y_gb":ih, "x_gb": x_last, "y_lc": y_lc, "x_lc": x_lc, 'size': size, "H": H, "W": W}
    sio.savemat(file_mat, data)
    index += 1  

  y_last = H - size
  for iw in range(0, W // size * size , size):
    file_mat = os.path.join(folder_dir, '{}.mat'.format(index))
    if iw == 0:
      crop_image = image[y_last-2*offset:y_last-2*offset + patch_size, iw:iw+patch_size]
      y_lc = 2*offset ; x_lc = 0
    else:
      crop_image = image[y_last-2*offset:y_last-2*offset + patch_size, iw-offset:iw-offset+patch_size]
      y_lc = 2*offset; x_lc = offset

    assert crop_image.shape[:2] == (patch_size, patch_size)
    data = {"image": crop_image, "y_gb":y_last, "x_gb": iw, "y_lc": y_lc, "x_lc": x_lc, 'size': size, "H": H, "W": W}
    sio.savemat(file_mat, data)
    index += 1  

  file_mat = os.path.join(folder_dir, '{}.mat'.format(index))
  crop_image = image[y_last-2*offset:, x_last-2*offset:]
  y_lc = 2*offset ; x_lc = 2*offset
  assert crop_image.shape[:2] == (patch_size, patch_size)
  data = {"image": crop_image, "y_gb":y_last, "x_gb": x_last, "y_lc": y_lc, "x_lc": x_lc, 'size': size, "H": H, "W": W}
  sio.savemat(file_mat, data)
  index += 1  

save_dir = '/root/codalab_split'
if os.path.exists(save_dir):
  shutil.rmtree(save_dir)
os.mkdir(save_dir)
for image_path in tqdm(glob.glob('/root/codalab_val/*.png')):
  crop_image(image_path, save_dir)


