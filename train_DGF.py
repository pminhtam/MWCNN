import torch
import argparse
# import utility
from model.mwcnn_dgf import MWCNN_DGF
from torch.utils.data import DataLoader
import loss
import os

# import h5py
from option import args
from data.data_provider import SingleLoader_DGF
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
import numpy as np
# import model
from torchsummary import summary
from utils.metric import calculate_psnr
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
# from collections import OrderedDict
from model.mwcnn_noise_estimate import MWCNN_noise


if __name__ == "__main__":

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    # checkpoint = utility.checkpoint(args)
    data_set = SingleLoader_DGF(noise_dir=args.noise_dir,gt_dir=args.gt_dir,image_size=args.image_size,burst_length=args.burst_length)
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )

    loss_func = loss.Loss(args,None)
    loss_func_i = loss.LossAnneal_i()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    # model = MWCNN_DGF(args).to(device)
    if args.model_type == "DGF":
        model = MWCNN_DGF().to(device)
    elif args.model_type == "noise":
        model = MWCNN_noise().to(device)
    else:
        print(" Model type not valid")
        exit()
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 4, 6, 8, 10, 12, 14, 16], 0.8)

    optimizer.zero_grad()
    global_step = 0
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.restart:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
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
            model.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    for epoch in range(start_epoch, args.epochs):
        for step, (image_noise_hr,image_noise_lr, image_gt_hr, image_gt_lr) in enumerate(data_loader):
            burst_noise = image_noise_lr.to(device)
            gt = image_gt_hr.to(device)
            image_gt_lr = image_gt_lr.to(device)
            image_noise_hr = image_noise_hr.to(device)
            pred_i, pred = model(burst_noise,image_noise_hr)
            # print(pred.size())
            loss_basic = loss_func(pred, gt)
            loss_i = loss_func_i(global_step, pred_i, image_gt_lr)
            loss = loss_basic + loss_i
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
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                print('{:-4d}\t| epoch {:2d}\t| step {:4d}\t| loss_basic: {:.4f}\t|'
                      ' loss: {:.4f}\t| PSNR: {:.2f}dB\t.'
                      .format(global_step, epoch, step, loss_basic, loss, calculate_psnr(pred, gt)))
                # print(global_step ,"PSNR  : ",calculate_psnr(pred,gt))
                print(average_loss.get_value())
            global_step +=1
        scheduler.step()

    # print(model)
    # print(summary(model,[(3,512,512),[8]]))