This is a tensorflow implement of SCAttNet: Semantic Segmentation Network with Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images

# Prerequisites

tensorflow 1.4.0
python 2.7

1. Please put your train image, train_label into the path ./dataset/train_img, ./dataset/train_label
2. Run python train.py to get the checkpoint file
3. Run python predict.py to get the predicted results.

# The manuscript:
Abstract: High-resolution remote sensing images (HRRSIs) contain substantial ground object information, such as texture, shape, and spatial location. Semantic segmentation, which is an important method for element extraction, has been widely used in processing mass HRRSIs. However, HRRSIs often exhibit large intraclass variance and small interclass variance due to the diversity and complexity of ground objects, thereby bringing great challenges to a semantic segmentation task. In this study, we propose a new end-to-end semantic segmentation network, which integrates two lightweight attention mechanisms that can refine features adaptively. We compare our method with several previous advanced networks on the ISPRS Vaihingen and Potsdam datasets. Experimental results show that our method can achieve better semantic segmentation results compared with other works.

The manuscript can be visited at https://ieeexplore.ieee.org/document/9081937 or https://arxiv.org/abs/1912.09121

If this repo is useful in your research, please kindly consider citing our paper as follow.
'''
Tex
Haifeng Li, Kaijian Qiu, Li Chen, Xiaoming Mei, Liang Hong, Chao Tao*. SCAttNet: Semantic Segmentation Network With Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images. IEEE Geoscience and Remote Sensing Letters. 2020:1-5. DOI: 10.1109/LGRS.2020.2988294 

Bibtex
@article{li2020SCAttNet,
    title={SCAttNet: Semantic Segmentation Network With Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images},
    author={Li, Haifeng and Qiu, Kaijian and Chen, Li and Mei, Xiaoming and Liang, Hong and Tao, Chao},
    journal={IEEE Geoscience and Remote Sensing Letters},
    DOI = {10.1109/LGRS.2020.2988294},
    year={2020},
    page={1-5}
    type = {Journal Article}
}

Endnote
%0 Journal Article
%A Li, Haifeng
%A Qiu, Kaijian
%A Chen, Li
%A Mei, Xiaoming
%A Liang, Hong
%A Tao, Chao
%D 2020
%T SCAttNet: Semantic Segmentation Network With Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images
%B IEEE Geoscience and Remote Sensing Letters
%R DOI:10.1109/LGRS.2020.2988294
%! SCAttNet: Semantic Segmentation Network With Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images
'''
