# MWCNN 

Source https://github.com/lpj-github-io/MWCNNv2

Paper [Multi-level Wavelet Convolutional Neural Networks](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8732332)

#Train

```
python train_eval_syn_DGF.py --noise_dir ../image/Noisy/ --gt_dir ../image/Clean/  \
--image_size 256 --batch_size 8 --save_every 1000 --loss_every 100   \
-nw 8  -ckpt checkpoints --resetart
```

# Test 
```
python test_mat.py  -n /mnt/vinai/SIDD/ValidationNoisyBlocksSrgb.mat  -g /mnt/vinai/SIDD/ValidationGtBlocksSrgb.mat \
  -ckpt checkpoints 

```

