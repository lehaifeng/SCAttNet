# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:21:39 2018

@author: lhf
"""
import numpy as np
import glob
import scipy
import matplotlib.pyplot as plt

def load_batch(x,y):
    x1=[]
    y1=[]
    for i in range(len(x)):
        img=scipy.misc.imread(x[i])/255.0
        lab=scipy.misc.imread(y[i])
        x1.append(img)
        y1.append(lab)
    y1=np.array(y1).astype(np.float32)   
    return x1,y1


def next_batch():
    img=np.array(sorted(glob.glob('./dataset/train_img/*.png')))
    label=np.array(sorted(glob.glob('./dataset/train_label/*.png')))
    index=np.random.choice(len(img),16)              
    x_batch=img[index]
    y_batch=label[index]
    image_batch,lab_batch=load_batch(x_batch,y_batch)
    return image_batch,lab_batch
    
def loadtestdata():
    img=sorted(glob.glob('./dataset/test_img/*.png'))
    label=sorted(glob.glob('./dataset/test_label/*.png'))
    
    
    train_img=[]
    for i in img:
        img1=scipy.misc.imread(i)/255.0
        train_img.append(img1)
    train_image=np.array(train_img).astype(np.float32)
    label_img=[]
    for i in label:
        label1=scipy.misc.imread(i)
       
        label_img.append(label1)
    label_image=np.array(label_img)
    
    return train_image,label_image
    

