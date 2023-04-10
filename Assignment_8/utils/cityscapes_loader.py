# the content of this file was mainly copied from  https://github.com/meetps/pytorch-semseg/blob/master/ptsemseg/loader/cityscapes_loader.py
# instead of using scipy.misc.imread and scipy.misc.resize, we use PIL.Image.open PIL.Image.resize as this functions are depricated since version 1.3.0
# https://docs.scipy.org/doc/scipy-1.2.1/reference/generated/scipy.misc.imresize.html

import os
import torch
import numpy as np
#import scipy.misc as m
import PIL.Image as Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from torch.utils import data

# add augmentation https://github.com/hoya012/semantic-segmentation-tutorial-pytorch/blob/master/learning/utils.py

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]


class cityscapesLoader(data.Dataset):
    """cityscapesLoader
    https://www.cityscapes-dataset.com
    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/
    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(
        self,
        root,
        split="train",
        is_transform=False,
        img_size=(256, 512),
        augmentations=None,
        use_default_aug=False,
        img_norm=True,
        version="cityscapes",
        test_mode=False,
        silent=False,
    ):
        """__init__
        :param root:
        :param split:
        :param is_transform:
        :param img_size:
        :param augmentations
        """
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.augmentations = augmentations
        self.use_default_aug = use_default_aug
        self.img_norm = img_norm
        self.n_classes = 19
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array(self.mean_rgb[version])
        self.files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(self.root, "gtFine", self.split)

        self.files[split] = recursive_glob(rootdir=self.images_base, suffix=".png")

        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [
            7,
            8,
            11,
            12,
            13,
            17,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            31,
            32,
            33,
        ]
        self.class_names = [
            "unlabelled",
            "road",
            "sidewalk",
            "building",
            "wall",
            "fence",
            "pole",
            "traffic_light",
            "traffic_sign",
            "vegetation",
            "terrain",
            "sky",
            "person",
            "rider",
            "car",
            "truck",
            "bus",
            "train",
            "motorcycle",
            "bicycle",
        ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if not self.files[split]:
            raise Exception("No files for split=[%s] found in %s" % (split, self.images_base))
        if not silent:
            print("Found %d %s images" % (len(self.files[split]), split))

    def __len__(self):
        """__len__"""
        return len(self.files[self.split])

    def __getitem__(self, index):
        """__getitem__
        :param index:
        """
        img_path = self.files[self.split][index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = Image.open(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = Image.open(lbl_path)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        if self.augmentations is not None:
            img, lbl = self.augmentations(img, lbl)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)

        return img, lbl

    def transform(self, img, lbl):
        """transform
        :param img:
        :param lbl:
        """
        #img = m.imresize(img, (self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
        img = Image.fromarray(img)
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img = np.array(img, dtype=np.uint8)
        #img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        if self.img_norm:
            # Resize scales images from 0 to 255, thus we need
            # to divide by 255.0
            img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = Image.fromarray(lbl)
        lbl = lbl.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        lbl = np.array(lbl, dtype=float)
        # add a channel dimension
        lbl = np.expand_dims(lbl, axis=0)
        lbl = lbl.astype(int)


        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        # ------------- Augmentation -------------
        if self.use_default_aug:

            # Generate random parameters for augmentation
            bf = np.random.uniform(0.8, 1.2)
            cf = np.random.uniform(0.8, 1.2)
            sf = np.random.uniform(0.8, 1.2)
            hf = np.random.uniform(-0.2, 0.2)
            pflip = np.random.randint(0,1) > 0.5
            

            # H-flip
            if pflip == True:
                img = TF.hflip(img)
                lbl = TF.hflip(lbl)

            # Color jitter
            # convert to pil image
            img = np.transpose(img, (1, 2, 0))
            img = (img*255).astype(np.uint8)
            img = TF.to_pil_image(img)
            img = TF.adjust_brightness(img, bf)
            img = TF.adjust_contrast(img, cf)
            img = TF.adjust_saturation(img, sf)
            img = TF.adjust_hue(img, hf)
            

            


            img = np.array(img, dtype=float) / 255.0
            # NHWC -> NCHW
            img = img.transpose(2, 0, 1)





        # -------------- End Augmentation --------------

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        crf = np.random.uniform(0, 1) # random crop with probability 0.3
        if(self.use_default_aug and crf > 0.3):

            # # Random crop
            i, j, h, w = transforms.RandomCrop.get_params(
                img, output_size=(self.img_size[0]//2, self.img_size[1]//2)
            )

            #print("i, j, h, w", i, j, h, w)


            img = TF.crop(img, i, j, h, w)  # (C, H, W)
            lbl = TF.crop(lbl, i, j, h, w)  # (1, H, W)
            # Resize to original size
            img = TF.resize(img, (self.img_size[0], self.img_size[1]))
            lbl = TF.resize(lbl, (self.img_size[0], self.img_size[1]), interpolation=transforms.InterpolationMode.NEAREST)

        # # Normalize
        #add normalization of the image
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img = TF.normalize(img, mean, std)

        return img, lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        for _voidc in self.void_classes:
            mask[mask == _voidc] = self.ignore_index
        for _validc in self.valid_classes:
            mask[mask == _validc] = self.class_map[_validc]
        return mask