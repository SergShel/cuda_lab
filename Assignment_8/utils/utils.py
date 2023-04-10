import torch  
import os
import datetime
import math


def save_model(model, optimizer, epoch, stats, model_name=""):
    """ Saving model checkpoint """
    dir_name = "checkpoints"
    if model_name != "":
        dir_name = f"{dir_name}_{model_name}"
    create_dir(dir_name)
    savepath = f"{dir_name}/checkpoint_epoch_{epoch}.pth"

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)
    return


def load_model(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    
    checkpoint = torch.load(savepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]
    
    return model, optimizer, epoch, stats

def create_dir(path):
    """ Creating directory if it does not exist already """
    if not os.path.exists(path):
        os.makedirs(path)
    return

def timestamp():
    """
    Obtaining the current timestamp in an human-readable way
    """
    timestamp = str(datetime.datetime.now()).split('.')[0].replace(' ', '_').replace(':', '-')
    return timestamp


def IoU(pred, target, num_classes):
    """ Computing the IoU for a single image """
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)

    for lbl in range(num_classes):
        pred_inds = pred == lbl
        target_inds = target == lbl
        
        intersection = (pred_inds[target_inds]).long().sum().cpu()
        union = pred_inds.long().sum().cpu() + target_inds.long().sum().cpu() - intersection
        iou = intersection / (union + 1e-8)
        iou = iou + 1e-8 if union > 1e-8 and not math.isnan(iou) else 0
        ious.append(iou)
    return torch.tensor(ious)