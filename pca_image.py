#!/usr/bin/python
# -*- coding: utf-8 -*-
from PIL import Image
from numpy import *
from pylab import *
def pca(X):
    num_data,dim = X.shape
    mean_X = X.mean(axis=0)
    XX = X - mean_X
    if dim > num_data:
    	M = dot(X,X.T)
    	e,EV = linalg.eigh(M) #特征值和特征向量
    	tmp = dot(X.T,EV).T
    	V = tmp[::-1]
    	S = sqrt(e)[::-1]
    	for i in range(V.shape[1]):
    		V[:,1]/=S
    else:
    	U,S,V=linalg.svd(X)
    	VV = V[:num_data]
    return V,S,mean_X
imlist = ['IMG_1629.jpg','IMG_1629.jpg','IMG_1629.jpg','IMG_1629.jpg','IMG_1629.jpg']
img = Image.open(imlist[0])
im = array(img)
m,n = im.shape[0:2]
#print m,n
imnbr = len(imlist)
immatrix = array([array(Image.open(im)).flatten()  
for im in imlist],'f')
V,S,immean = pca(immatrix);
plt.figure()
plt.gray()
#subplot(2,4,1)
plt.imshow(img)
plt.imshow(immean.reshape(m,n,-1))
plt.show()