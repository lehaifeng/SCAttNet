# -*- coding: utf-8 -*-
"""
Created on Mon Jul 23 10:21:39 2018

@author: lhf
"""
import numpy as np
import glob
import scipy.misc
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
    print(y1.shape)
    return x1,y1


def next_batch():
    img=np.array(sorted(glob.glob('./dataset/train_img/*.png')))
#    print(img)
    label=np.array(sorted(glob.glob('./dataset/train_label/*.png')))
#    print(label)
    index=np.random.choice(len(img),16)
#    print(index)
#    print(index)                  
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
    
a,b=next_batch()

#print(a)
#for i in range(5):
#    plt.imshow(a[i])
#    plt.show()
#    plt.imshow(np.squeeze(b[i]))
#    plt.show()

#c,d=loaddata()
###############load train###############
#def loaddata():
#    img=glob.glob('./datasets/train/img/*.png')
#    label=glob.glob('./datasets/train/gt/*.png')
#    img_list=[]
#    label_list=[]
#    for i in img:
#        img1=scipy.misc.imread(i)
#        
#        img_list.append(img1)
#    train_img=np.array(img_list)/255.0
#        
#    for i in label:
#        label1=scipy.misc.imread(i)
#        label_list.append(label1)
#    label_img=np.array(label_list)
#    
#    ##########load text##########
#    img=glob.glob('./datasets/val/img/*.png')
#    label=glob.glob('./datasets/val/gt/*.png')
#    img_list=[]
#    label_list=[]
#    for i in img:
#        img1=scipy.misc.imread(i)
#        
#        img_list.append(img1)
#    test_img=np.array(img_list)/255.0
#        
#    for i in label:
#        label1=scipy.misc.imread(i)
#        label_list.append(label1)
#    test_label=np.array(label_list)
#    
#    return train_img,label_img,test_img,test_label
#a,b=loaddata()
#for i in range(10):
#    plt.imshow(a[i])
#    plt.show()
#    plt.imshow(np.squeeze(b[i]))
#    plt.show()
##plt.figure(dpi=200)
##plt.subplot(221)
#plt.imshow(a[0])
#plt.subplot(222)
#plt.imshow(b[0])
#
#plt.subplot(223)
#plt.imshow(a[1])
#plt.subplot(224)
#plt.imshow(b[1])


#plt.subplot(525)
#plt.imshow(a[2])
#plt.subplot(526)
#plt.imshow(b[2])
#
#plt.subplot(527)
#plt.imshow(a[3])
#plt.subplot(528)
#plt.imshow(b[3])
#
#plt.subplot(529)
#plt.imshow(a[4])
#plt.subplot(52,10)
#plt.imshow(b[4])
