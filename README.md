# EAR-NET: Error Attention Refining Network ForRetinal Vessel Segmentation

PyTorch implementation of 

[EAR-NET: Error Attention Refining Network ForRetinal Vessel Segmentation](https://arxiv.org/pdf/2107.01351.pdf)" ( DICTA 2021 ) 

If you use the code in this repo for your work, please cite the following bib entries:

    @article{wang2021ear,
        title={EAR-NET: Error Attention Refining Network For Retinal Vessel Segmentation},
        author={Wang, Jun and Yang, Zhao and Qian, Linglong and Yu, Xiaohan and Gao, Yongsheng},
        journal={arXiv preprint arXiv:2107.01351},
        eprint={2107.01351},
        archivePrefix={arXiv},
        primaryClass={cs.CV}
    }


## Abstract

The precise detection of blood vessels in retinal images is crucial to the early diagnosis of the retinal vascular diseases, e.g., diabetic, hypertensive and solar retinopathies. Existing works often fail in predicting the abnormal areas, e.g, sudden brighter and darker areas and are inclined to predict a pixel to background due to the significant class imbalance, leading to high accuracy and specificity while low sensitivity. To that end, we propose a novel error attention refining network (ERA-Net) that is capable of learning and predicting the potential false predictions in a two-stage manner for effective retinal vessel segmentation. The proposed ERA-Net in the refine stage drives the model to focus on and refine the segmentation errors produced in the initial training stage. To achieve this, unlike most previous attention approaches that run in an unsupervised manner, we introduce a novel error attention mechanism which considers the differences between the ground truth and the initial segmentation masks as the ground truth to supervise the attention map learning. Experimental results demonstrate that our method achieves state-of-the-art performance on two common retinal blood vessel datasets.

<img src='architecture.png' width='1280' height='350'>


## Prerequisites

The following packages are required to run the scripts:
- [Python >= 3.6]
- [PyTorch >= 1.0]
- [Torchvision]

## Dataset
The dataset can be downloaded from [Google Drive](https://drive.google.com/drive/folders/1EF_iamMlnb0QYS2xiQRq--fxm7an4tv7?usp=sharing).

soybean_1_1: The soycultivarvein dataset with the training set:test set=1:1. For comparison to the state-of-the-art hand-crafted methods.

soybean_2_1: The soycultivarvein dataset with the training set:test set=2:1.


## Training scripts for MGANet with the backbone network Densenet161.
Train the model on the Soybean dataset. We run our experiments on 4x2080Ti/4x1080Ti with the batchsize of 32.

    $ python train.py --dataset soybean_2_1 --lr 0.05 --backbone_class densenet161


## Testing scripts for MGANet with the backbone network Densenet161.
Test the model on the Soybean dataset:

    $ python test.py  --dataset soybean_2_1 --backbone_class densenet161
    
        
            
## Download  Models


[Trained model Google Drive](https://drive.google.com/drive/folders/11SA7PGR9NbyJEaXFOHwA_PGiORdIEoYZ?usp=sharing)

## Segmentation Experiments.
For the leaf vein segmentation experiments, please refer to [Nvidia/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation) to gain the details.

All the three datasets are trained with the crop size (448,448) and 60 epochs.



## Acknowledgment
Thanks for the advice and guidance given by Dr.Xiaohan Yu and Prof. Yongsheng Gao.

Our project references the codes in the following repos.
- [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
- [Nvidia/semantic-segmentation](https://github.com/NVIDIA/semantic-segmentation)














### [Paper](https://arxiv.org/abs/2005.10821) | [YouTube](https://youtu.be/odAGA7pFBGA)  | [Cityscapes Score](https://www.cityscapes-dataset.com/method-details/?submissionID=7836) <br>

Pytorch implementation of our paper [Hierarchical Multi-Scale Attention for Semantic Segmentation](https://arxiv.org/abs/2005.10821).<br>

Please refer to the `sdcnet` branch if you are looking for the code corresponding to [Improving Semantic Segmentation via Video Prediction and Label Relaxation](https://nv-adlr.github.io/publication/2018-Segmentation).

## Installation 

* The code is tested with pytorch 1.3 and python 3.6
* You can use ./Dockerfile to build an image.


## Download Weights

* Create a directory where you can keep large files. Ideally, not in this directory.
```bash
  > mkdir <large_asset_dir>
```

* Update `__C.ASSETS_PATH` in `config.py` to point at that directory

  __C.ASSETS_PATH=<large_asset_dir>

* Download pretrained weights from [google drive](https://drive.google.com/open?id=1fs-uLzXvmsISbS635eRZCc5uzQdBIZ_U) and put into `<large_asset_dir>/seg_weights`

## Download/Prepare Data

If using Cityscapes, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_DIR=<path_to_cityscapes>
```

* Download Autolabelled-Data from [google drive](https://drive.google.com/file/d/1DtPo-WP-hjaOwsbj6ZxTtOo_7R_4TKRG/view?usp=sharing)

If using Cityscapes Autolabelled Images, download Cityscapes data, then update `config.py` to set the path:
```python
__C.DATASET.CITYSCAPES_CUSTOMCOARSE=<path_to_cityscapes>
```

If using Mapillary, download Mapillary data, then update `config.py` to set the path:
```python
__C.DATASET.MAPILLARY_DIR=<path_to_mapillary>
```


## Running the code

The instructions below make use of a tool called `runx`, which we find useful to help automate experiment running and summarization. For more information about this tool, please see [runx](https://github.com/NVIDIA/runx).
In general, you can either use the runx-style commandlines shown below. Or you can call `python train.py <args ...>` directly if you like.


### Run inference on Cityscapes

Dry run:
```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i -n
```
This will just print out the command but not run. It's a good way to inspect the commandline. 

Real run:
```bash
> python -m runx.runx scripts/eval_cityscapes.yml -i
```

The reported IOU should be 86.92. This evaluates with scales of 0.5, 1.0. and 2.0. You will find evaluation results in ./logs/eval_cityscapes/...

### Run inference on Mapillary

```bash
> python -m runx.runx scripts/eval_mapillary.yml -i
```

The reported IOU should be 61.05. Note that this must be run on a 32GB node and the use of 'O3' mode for amp is critical in order to avoid GPU out of memory. Results in logs/eval_mapillary/...

### Dump images for Cityscapes

```bash
> python -m runx.runx scripts/dump_cityscapes.yml -i
```

This will dump network output and composited images from running evaluation with the Cityscapes validation set. 

### Run inference and dump images on a folder of images

```bash
> python -m runx.runx scripts/dump_folder.yml -i
```

You should end up seeing images that look like the following:

![alt text](imgs/composited_sf.png "example inference, composited")

## Train a model

Train cityscapes, using HRNet + OCR + multi-scale attention with fine data and mapillary-pretrained model
```bash
> python -m runx.runx scripts/train_cityscapes.yml -i
```

The first time this command is run, a centroid file has to be built for the dataset. It'll take about 10 minutes. The centroid file is used during training to know how to sample from the dataset in a class-uniform way.

This training run should deliver a model that achieves 84.7 IOU.

## Train SOTA default train-val split
```bash
> python -m runx.runx  scripts/train_cityscapes_sota.yml -i
```
Again, use `-n` to do a dry run and just print out the command. This should result in a model with 86.8 IOU. If you run out of memory, try to lower the crop size or turn off rmi_loss.
