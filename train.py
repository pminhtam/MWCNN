import torch
import argparse
import utility
import model
from torch.utils.data import DataLoader
import loss
import os

# import h5py
from option import args
from data.data_provider import SingleLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
# import model
from torchsummary import summary
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
# from collections import OrderedDict


if __name__ == "__main__":

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    data_set = SingleLoader(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size)
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.n_threads
    )

    loss_func = loss.Loss(args, checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = "checkpoint/"
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model = model.Model(args).to(device)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    optimizer.zero_grad()
    global_step = 0
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_iter']
        best_loss = checkpoint['best_loss']
        state_dict = checkpoint['state_dict']
        # new_state_dict = OrderedDict()
        # for k, v in state_dict.items():
        #     name = "model."+ k  # remove `module.`
        #     new_state_dict[name] = v
        model.model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
    except:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    for epoch in range(start_epoch, args.epochs):
        for step, (noise, gt) in enumerate(data_loader):
            noise = noise.to(device)
            gt = gt.to(device)
            pred = model(noise,0)
            # print(pred.size())
            loss = loss_func(pred,gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            average_loss.update(loss)
            if global_step % args.save_every == 0:
                print(len(average_loss._cache))
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                print(global_step ,"PSNR  : ",calculate_psnr(pred,gt))
                print(average_loss.get_value())
            global_step +=1

    # print(model)
    print(summary(model,[(3,512,512),[8]]))