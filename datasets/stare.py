"""
Copyright 2020 Nvidia Corporation

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
   may be used to endorse or promote products derived from this software
   without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""
import os
import os.path as path

from config import cfg
from runx.logx import logx
from datasets.base_loader import BaseLoader
import datasets.stare_labels as stare_labels
import datasets.uniform as uniform
from datasets.utils import make_dataset_folder

def train_val_split(root, split, cv_split):
    """
    Find cities that correspond to a given split of the data. We split the data
    such that a given city belongs to either train or val, but never both. cv0
    is defined to be the default split.

     all_cities = [x x x x x x x x x x x x]
     val:
       split0     [x x x                  ]
       split1     [        x x x          ]
       split2     [                x x x  ]
     trn:
       split0     [      x x x x x x x x x]
       split1     [x x x x       x x x x x]
       split2     [x x x x x x x x        ]

    split - train/val/test
    cv_split - 0,1,2,3

    cv_split == 3 means use train + val
    """
    filenames=os.listdir(path.join(root,'train','images'))
    trn_imgs=sorted(filenames)
    
    filenames=os.listdir(path.join(root,'validate','images'))
    val_imgs=sorted(filenames)
    
    print('val_imgs',val_imgs)       

    #val_imgs=list(range(1,401))+list(range(601,1001))

    #all_cities = val_cities + trn_cities

    return trn_imgs if split=='train' else val_imgs


class Loader(BaseLoader):
    num_classes = 2
    ignore_label = 255
    trainid_to_name = {}
    color_mapping = []

    def __init__(self, mode, quality='fine', joint_transform_list=None,
                 img_transform=None, label_transform=None, eval_folder=None):

        super(Loader, self).__init__(quality=quality, mode=mode,
                                     joint_transform_list=joint_transform_list,
                                     img_transform=img_transform,
                                     label_transform=label_transform)

        ######################################################################
        # Cityscapes-specific stuff:
        ######################################################################
        self.root = cfg.DATASET.STARE_DIR
        self.id_to_trainid = stare_labels.label2trainid
        self.trainid_to_name = stare_labels.trainId2name
        self.fill_colormap()
        img_ext = '.ppm'
        mask_ext = '.ppm'
        if mode=='train':
            img_root = path.join(self.root, 'train','images')
            mask_root = path.join(self.root,'train','labels')
            if cfg.MODEL.REFINEMENT:
                error_root = path.join(self.root,'train','errors')
            else:
                error_root=None     
        else:
            img_root = path.join(self.root, 'validate','images')
            mask_root = path.join(self.root,'validate','labels')
            error_root=None
        if mode == 'folder':
            self.all_imgs = make_dataset_folder(eval_folder)
        else:
            self.filenames = train_val_split(self.root, mode, cfg.DATASET.CV)
            self.all_imgs = self.find_all_images(self.filenames,img_root, mask_root,error_root, img_ext, mask_ext)

        logx.msg(f'cn num_classes {self.num_classes}')
        self.fine_centroids = uniform.build_centroids(self.all_imgs,
                                                      self.num_classes,
                                                      self.train,
                                                      cv=None,
                                                      id2trainid=self.id_to_trainid)
        self.centroids = self.fine_centroids
        self.build_epoch()

    def disable_coarse(self):
        """
        Turn off using coarse images in training
        """
        self.centroids = self.fine_centroids

    def only_coarse(self):
        """
        Turn on using coarse images in training
        """
        print('==============+Running Only Coarse+===============')
        self.centroids = self.coarse_centroids

    def find_all_images(self, filenames, img_root, mask_root,error_root, img_ext,
                               mask_ext):
        """
        Find image and segmentation mask files and return a list of
        tuples of them.

        Inputs:
        img_root: path to parent directory of train/val/test dirs
        mask_root: path to parent directory of train/val/test dirs
        img_ext: image file extension
        mask_ext: mask file extension
        cities: a list of cities, each element in the form of 'train/a_city'
          or 'val/a_city', for example.
        """
        items = []
        for filename in filenames:
            full_img_fn = os.path.join(img_root, filename)
            full_mask_fn = os.path.join(mask_root, filename)
            if error_root is not None:
                error_map_fn=os.path.join(error_root, filename)
                items.append((full_img_fn, full_mask_fn, error_map_fn))
            else:
                items.append((full_img_fn, full_mask_fn,None))

        logx.msg('mode {} found {} images'.format(self.mode, len(items)))

        return items

    def fill_colormap(self):
        '''
        palette = [128, 64, 128,
                   244, 35, 232,
                   70, 70, 70,
                   220,220,220,
                   190, 153, 153,
                   153, 153, 153,
                   250, 170, 30,
                   220, 220, 0,
                   107, 142, 35,
                   152, 251, 152,
                   70, 130, 180,
                   220, 20, 60,
                   255, 0, 0,
                   0, 0, 142,
                   0, 0, 70,
                   0, 60, 100,
                   0, 80, 100,
                   0, 0, 230,
                   119, 11, 32]
        '''
        palette = [0,0,0,
                   255, 255, 255,
                   255,0,0,
                   0,0,255,
                   190, 153, 153,
                   153, 153, 153,
                   250, 170, 30,
                   220, 220, 0,
                   107, 142, 35,
                   152, 251, 152,
                   70, 130, 180,
                   220, 20, 60,
                   255, 0, 0,
                   0, 0, 142,
                   0, 0, 70,
                   0, 60, 100,
                   0, 80, 100,
                   0, 0, 230,
                   119, 11, 32]
        zero_pad = 256 * 3 - len(palette)
        for i in range(zero_pad):
            palette.append(0)
        self.color_mapping = palette
