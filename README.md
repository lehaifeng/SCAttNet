# SCAttNet
Semantic Segmentation Network with Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images

# The manuscript:
SCAttNet: Semantic Segmentation Network with Spatial and Channel Attention Mechanism for High-Resolution Remote Sensing Images

Abstract: High-resolution remote sensing images (HRRSIs) contain substantial ground object information, such as texture, shape, and spatial location. Semantic segmentation, which is an important method for element extraction, has been widely used in processing mass HRRSIs. However, HRRSIs often exhibit large intraclass variance and small interclass variance due to the diversity and complexity of ground objects, thereby bringing great challenges to a semantic segmentation task. In this study, we propose a new end-to-end semantic segmentation network, which integrates two lightweight attention mechanisms that can refine features adaptively. We compare our method with several previous advanced networks on the ISPRS Vaihingen and Potsdam datasets. Experimental results show that our method can achieve better semantic segmentation results compared with other works.

https://arxiv.org/abs/1912.09121

# Prerequisites

tensorflow 1.4.0
python 2.7

1. Please put your train image, train_label into the path ./dataset/train_img, ./dataset/train_label
2. Run python train.py to get the checkpoint file
3. Run python predict.py to get the predicted results.




