import torch
from torch.utils.tensorboard import SummaryWriter
import utils.utils as utils
import utils.visualizations as vis
from tqdm import tqdm
import numpy as np
from utils.augmentation import copyblob, rand_bbox
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def train_model(model, optimizer, scheduler, criterion, train_loader, valid_loader,
                num_epochs, tboard=None, start_epoch=0, copy_blob=False, cut_mix=False, device="cuda", writer:SummaryWriter=None, model_name=""):
    """
    Training a model for a given number of epochs
    """
    
    for epoch in range(num_epochs):       
        # validation epoch
        model.eval()  # important for dropout and batch norms
        loss, (mAcc, mIoU, dice_coeff) = eval_model(
                model=model,
                eval_loader=valid_loader,
                criterion=criterion,
                device=device,
                epoch=epoch + start_epoch,
                writer=writer
            )
        writer.add_scalar(f'Metrics/Valid mAcc', mAcc, global_step=epoch+start_epoch)
        writer.add_scalar(f'Metrics/Valid mIoU', mIoU, global_step=epoch+start_epoch)
        writer.add_scalar(f'Loss/Valid', loss, global_step=epoch+start_epoch)
        writer.add_scalar(f'Metrics/Valid Dice_Coeff', dice_coeff, global_step=epoch+start_epoch)
        
        # training epoch
        model.train()  # important for dropout and batch norms
        mean_loss, cur_loss_iters = train_epoch(
                model=model,
                train_loader=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                epoch=epoch + start_epoch,
                device=device,
                writer=writer,
                copy_blob=copy_blob,
                cut_mix=cut_mix
            )
        scheduler.step(loss)
        writer.add_scalar(f'Loss/Train', mean_loss, global_step=epoch+start_epoch)
        writer.add_scalars(
                f'Loss/Combined',
                {"train_loss": mean_loss, "valid_loss": loss},
                global_step=epoch+start_epoch
            )
            
        if epoch % 3 == 0:
            utils.save_model(model, optimizer, epoch + start_epoch, stats={}, model_name=model_name)
        
    print(f"Training completed")
    return


def train_epoch(model, train_loader, optimizer, criterion, epoch, device, writer, copy_blob, cut_mix):
    """ Training a model for one epoch """
    
    loss_list = []
    for i, (images, labels) in enumerate(tqdm(train_loader)):

        # --------------------- Augmentation ---------------------
        if copy_blob:
            for i in range(images.size()[0]):
                rand_idx = np.random.randint(images.size()[0])
                # wall(3) --> sidewalk(1)
                copyblob(src_img=images[i], src_mask=labels[i], dst_img=images[rand_idx], dst_mask=labels[rand_idx], src_class=3, dst_class=1)
                # fence(4) --> sidewalk(1)
                copyblob(src_img=images[i], src_mask=labels[i], dst_img=images[rand_idx], dst_mask=labels[rand_idx], src_class=4, dst_class=1)
                # bus(15) --> road(0)
                copyblob(src_img=images[i], src_mask=labels[i], dst_img=images[rand_idx], dst_mask=labels[rand_idx], src_class=15, dst_class=0)
                # train(16) --> road(0)
                copyblob(src_img=images[i], src_mask=labels[i], dst_img=images[rand_idx], dst_mask=labels[rand_idx], src_class=16, dst_class=0)

        
        # generate uniform distribution from 0 to 1
        prob = np.random.rand(1)
        cutmix_prob = 0.5 # probability of applying cutmix

        if cut_mix and prob < cutmix_prob:
            # generate mixed sample
                lam = np.random.beta(1., 1.)
                rand_index = torch.randperm(images.size()[0])
                bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                images[:, :, bbx1:bbx2, bby1:bby2] = images[rand_index, :, bbx1:bbx2, bby1:bby2]
                labels[:, :, bbx1:bbx2, bby1:bby2] = labels[rand_index, :, bbx1:bbx2, bby1:bby2]
        # ------------------- End Augmentation -------------------


        images = images.to(device)
        labels = labels.long().to(device)
        labels[labels == 250] = 0
        
        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()
         
        # Forward pass to get output/logits
        outputs = model(images)

        # Calculate Loss: softmax --> cross entropy loss
        loss = criterion(outputs, labels[:, 0])
        loss_list.append(loss.item())
         
        # Getting gradients w.r.t. parameters
        loss.backward()
         
        # Updating parameters
        optimizer.step()
        
        if i % 30 == 0:
            iter_ = epoch * len(train_loader) + i
            lr = optimizer.param_groups[0]['lr']
            writer.add_scalar(f'Loss/Train Iters', loss.item(), global_step=iter_)
            writer.add_scalar(f'_Params/Learning Rate', lr, global_step=iter_)
        
    mean_loss = np.mean(loss_list)
    return mean_loss, loss_list

@torch.no_grad()
def eval_model(model, eval_loader, criterion, device, epoch, writer):
    """ Evaluating the model for either validation or test """
    correct_pixels = 0
    total_pixels = 0
    ious = []
    loss_list = []
    dice_coeff_list = []
    dice_coeff = DiceCoeff()
    
    for images, labels in tqdm(eval_loader):
        images = images.to(device)
        labels = labels.long().to(device)
        labels[labels == 250] = 0
        outputs = model(images)   
        loss = criterion(outputs, labels[:, 0])
        loss_list.append(loss.item())
        curr_dice_coeff = dice_coeff(outputs, labels[:, 0])
        dice_coeff_list.append(curr_dice_coeff.cpu().numpy())
        
        # computing evaluation metrics
        predicted_class = torch.argmax(outputs, dim=1)
        correct_pixels += predicted_class.eq(labels).sum().item()
        total_pixels += labels.numel()
        iou = utils.IoU(predicted_class, labels, num_classes=outputs.shape[1])
        ious.append(iou)
    
    # mean metrics and loss
    loss = np.mean(loss_list)
    dice_coeff = np.mean(dice_coeff_list)
    avg_accuracy = 100 * correct_pixels / total_pixels   
    ious = torch.stack(ious)
    ious = ious.sum(dim=-1) / (ious != 0).sum(dim=-1)  # per class IoU
    mIoU = ious.mean()  # averaging across classes
    # creating a visualization for tensorboard
    add_visualization(model, eval_loader, epoch, writer)
    return loss, (avg_accuracy, mIoU, dice_coeff)


@torch.no_grad()
def add_visualization(model, eval_loader, epoch, writer, device="cuda"):
    """ """
    imgs_arr, lbls_arr, preds_arr = [], [], []
    for i, (img, lbl) in enumerate(eval_loader):
        img = img.to(device)
        lbl[lbl == 250] = 0

        outputs = model(img) 
        _, preds = torch.max(outputs, 1)  
        
        imgs_arr.append(img.cpu())
        lbls_arr.append(lbl.cpu())
        #preds_arr.append(preds.cpu())
        break
    imgs_arr = torch.cat(imgs_arr, dim=0)
    lbls_arr = torch.cat(lbls_arr, dim=0)
    preds_arr = preds

    fig, ax = vis.qualitative_evaluation(imgs_arr, lbls_arr, preds_arr)
    writer.add_figure("Qualitative Eval", fig, global_step=epoch)
    return


"""
====================
Focal Loss
code reference: https://github.com/clcarwin/focal_loss_pytorch
====================
"""

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()



class DiceLoss(nn.Module):
    # https://github.com/shuaizzZ/Dice-Loss-PyTorch/blob/master/dice_loss.py
    """Dice Loss PyTorch
        Created by: Zhang Shuai
        Email: shuaizzz666@gmail.com
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss
    """
    def __init__(self, weight=None):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight) # Normalized weight
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss
    

class DiceCoeff(nn.Module):
    # inspired by DiceLoss function
    def __init__(self):
        super(DiceCoeff, self).__init__()
        self.smooth = 1e-5

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1) # (N, C, *)
        target = target.view(N, 1, -1) # (N, 1, *)

        predict = F.softmax(predict, dim=1) # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        target_onehot = torch.zeros(predict.size()).cuda()  # (N, 1, *) ==> (N, C, *)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)

        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / (union + self.smooth)  # (N, C)

        return dice_coef



"""
====================
Loss Function
====================
"""

def get_criterion(loss_type="ce", focal_gamma=0.5):
    # Define loss, optimizer and scheduler
    if loss_type == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=250)
    elif loss_type == 'weighted_ce':
        # Class-Weighted loss
        class_weight = [0.8373, 0.918, 0.866, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]
        class_weight.append(0) #for void-class
        class_weight = torch.FloatTensor(class_weight).cuda()
        criterion = nn.CrossEntropyLoss(weight=class_weight, ignore_index=250)
    elif loss_type =='focal':
        criterion = FocalLoss(gamma=focal_gamma)
    elif loss_type == 'dice':
        criterion = DiceLoss()
    else:
        raise NameError('Loss is not defined!')

    return criterion


