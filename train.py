import torch
import argparse
import utility
import model
from torch.utils.data import DataLoader
import loss

# import h5py
from option import args
from data.data_provider import SingleLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn as nn
# import model
from torchsummary import summary
from utils.metric import calculate_psnr

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
    model = model.Model(args)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    optimizer.zero_grad()
    global_step = 0
    for epoch in range(args.epochs):
        for step, (noise, gt) in enumerate(data_loader):
            pred = model(noise,0)
            # print(pred.size())
            loss = loss_func(pred,gt)
            if global_step % args.save_every == 0:
                torch.save(model.state_dict(), 'experiment/checkpoint{}.pth'.format(global_step))
            if global_step % args.loss_every == 0:
                print(global_step ,"PSNR  : ",calculate_psnr(pred,gt))
                print(loss)
            global_step +=1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    # print(model)
    print(summary(model,[(3,512,512),[8]]))