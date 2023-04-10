import os
import sys
sys.path.insert(0, "../utils")
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torchvision import datasets, models, transforms
import torchvision

from torch.utils.tensorboard import SummaryWriter
import shutil

plt.style.use('seaborn')
import seaborn as sns
sns.set_style("whitegrid", {'axes.grid' : False})


import helpers as h
import utils.utils as utils
import utils.visualizations as vis
import utils.cityscapes_loader as cityscapes_loader
import utils.train_eval as train_eval
from utils.train_eval import get_criterion
from utils.models import ResNetUNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def main():
    # load data
    dataset_root_dir = "/home/nfs/inf6/data/datasets/cityscapes/"

    train_ds = cityscapes_loader.cityscapesLoader(root=dataset_root_dir, split='train', is_transform=True)
    val_ds = cityscapes_loader.cityscapesLoader(root=dataset_root_dir, split='val', is_transform=True)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=12, shuffle=True, num_workers=4, drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=12, shuffle=False, num_workers=4, drop_last=True)

    # create model
    unet_res = ResNetUNet(n_class=19).to(device)
    # prepare optimizer, criterion and scheduler
    optimizer_unet_res = torch.optim.Adam(unet_res.parameters(), lr=0.001)
    # criterion_unet_res = get_criterion(loss_type="focal", focal_gamma=0.8)
    criterion_unet_res = get_criterion(loss_type="dice")
    scheduler_unet_res = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet_res, mode="min", factor=0.01, patience=10)
    # create tensorboard writer
    TBOARD_LOGS_unet_res = os.path.join(os.getcwd(), "tboard_logs/unet_res_dice", utils.timestamp())
    utils.create_dir(TBOARD_LOGS_unet_res)

    writer_unet_res = SummaryWriter(TBOARD_LOGS_unet_res)

    # freeze backbone (for 7 epochs)
    for param in unet_res.base_model.parameters():
        param.requires_grad = False

    train_eval.train_model(unet_res, optimizer_unet_res, scheduler_unet_res, criterion_unet_res, train_loader, val_loader,
                num_epochs=5, tboard=None, start_epoch=0, device="cuda", writer=writer_unet_res, copy_blob=True, cut_mix=True, model_name="unet_res_dice")

    # unfreeze backbone 
    for params in unet_res.layer0.parameters():
        param.requires_grad = True

    train_eval.train_model(unet_res, optimizer_unet_res, scheduler_unet_res, criterion_unet_res, train_loader, val_loader,
                num_epochs=50, tboard=None, start_epoch=5, device="cuda", writer=writer_unet_res, copy_blob=True, cut_mix=True, model_name="unet_res_dice")

if __name__ == "__main__":
    main()