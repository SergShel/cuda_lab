from matplotlib import pyplot as plt
import numpy as np
# import cityscapes_loader as cityscapes
from utils.cityscapes_loader import cityscapesLoader


"""
====================
Custom copyblob function for copyblob data augmentation
====================
"""

def copyblob(src_img, src_mask, dst_img, dst_mask, src_class, dst_class):
    dataset_root_dir = "/home/nfs/inf6/data/datasets/cityscapes/"
    City = cityscapesLoader(dataset_root_dir, silent=True)

    src_mask = src_mask.squeeze(axis=0)
    dst_mask = dst_mask.squeeze(axis=0)

    mask_hist_src, _ = np.histogram(src_mask.cpu().numpy().ravel(), len(City.valid_classes)-1, [0, len(City.valid_classes)-1])
    mask_hist_dst, _ = np.histogram(dst_mask.cpu().numpy().ravel(), len(City.valid_classes)-1, [0, len(City.valid_classes)-1])

    if mask_hist_src[src_class] != 0 and mask_hist_dst[dst_class] != 0:
        """ copy src blob and paste to any dst blob"""

        mask_y, mask_x = src_mask.size()
        """ get src object's min index"""
        src_idx = np.where(src_mask.cpu().numpy()==src_class)
        src_idx_sum = list(src_idx[0][i] + src_idx[1][i] for i in range(len(src_idx[0])))
        src_idx_sum_min_idx = np.argmin(src_idx_sum)        
        src_idx_min = src_idx[0][src_idx_sum_min_idx], src_idx[1][src_idx_sum_min_idx]
        
        """ get dst object's random index"""
        dst_idx = np.where(dst_mask.cpu().numpy()==dst_class)
        rand_idx = np.random.randint(len(dst_idx[0]))
        target_pos = dst_idx[0][rand_idx], dst_idx[1][rand_idx] 
        
        src_dst_offset = tuple(map(lambda x, y: x - y, src_idx_min, target_pos))
        dst_idx = tuple(map(lambda x, y: x - y, src_idx, src_dst_offset))
        
        for i in range(len(dst_idx[0])):
            dst_idx[0][i] = (min(dst_idx[0][i], mask_y-1))
        for i in range(len(dst_idx[1])):
            dst_idx[1][i] = (min(dst_idx[1][i], mask_x-1))
        try:
            dst_mask[dst_idx] = src_class
            dst_img[:, dst_idx[0], dst_idx[1]] = src_img[:, src_idx[0], src_idx[1]]
        except:
            pass
        # dst_mask[dst_idx] = src_class
        # dst_img[:, dst_idx[0], dst_idx[1]] = src_img[:, src_idx[0], src_idx[1]]


"""
====================
random bbox function for cutmix
====================
"""

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
