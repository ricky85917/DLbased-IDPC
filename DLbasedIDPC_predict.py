'''
############################################

2021-04-17

Created by AN-CIN, LI
Institute of Medical Device and Imaging
National Taiwan University, college of medicine

This work applied U-net for DL-based Isotropic Quantitative Differential Phase Contrast Microscopy

############################################
'''

import tensorflow as tf
import keras
import numpy as np 
import os
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Dropout, Conv2DTranspose
import matplotlib.pyplot as plt
from patches import extract_patches
import scipy.io
from itertools import product
from matplotlib.pyplot import imsave


def unet(pretrained_weights = None, input_size = (512,512,1)):
    inputs = Input(input_size)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', )(inputs)
    conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', )(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', )(pool1)
    conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', )(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', )(pool2)
    conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', )(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', )(pool3)
    conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', )(conv4)
    drop4 = Dropout(0.4)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', )(pool4)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', )(conv5)
    drop5 = Dropout(0.4)(conv5)
    
    up6 = Conv2DTranspose(128, 3, activation = 'relu', strides=2, padding='same', dilation_rate=2)(drop5)
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', )(merge6)
    conv6 = Conv2D(128, 3, activation = 'relu', padding = 'same', )(conv6)
    up7 = Conv2DTranspose(64, 3, activation = 'relu', strides=2, padding='same', dilation_rate=2)(conv6)
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', )(merge7)
    conv7 = Conv2D(64, 3, activation = 'relu', padding = 'same', )(conv7)
    up8 = Conv2DTranspose(32, 3, activation = 'relu', strides=2, padding='same', dilation_rate=2)(conv7)
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', )(merge8)
    conv8 = Conv2D(32, 3, activation = 'relu', padding = 'same', )(conv8)
    up9 = Conv2DTranspose(16, 3, activation = 'relu', strides=2, padding='same', dilation_rate=2)(conv8)
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', )(merge9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', )(conv9)
    conv10 = Conv2D(1, 1, activation = 'tanh')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-3), loss = 'mean_squared_error')
    
    if(pretrained_weights):
    	model.load_weights(pretrained_weights)

    return model


def resultEval(predict, groundtruth):
    metricsname = model.metrics_names
    result = model.evaluate(predict, groundtruth)
    for ii in range(len(metricsname)):
        try:
            print(metricsname[ii] + ':' + str(result[ii]) + '\n')
        except:
            print(metricsname[ii] + ':' + str(result) + '\n')

def normalizeidpc(xt,yt,totalmax,totalmin):
    interval = totalmax-totalmin
    xt = (xt-totalmin)/interval
    yt = (yt-totalmin)/interval
    return xt,yt

def idpcimage(oneaxis, twelve, hight, width, patchwidth):
    tifdir = os.listdir(oneaxis)
    tifdir2 = os.listdir(twelve)
    
    tifdir.sort(key=lambda x:(x.split('_')[0],(int(x.split('_')[-1].split('.')[0]))))
    tifdir2.sort(key=lambda x:(x.split('_')[0],(int(x.split('_')[-1].split('.')[0]))))
    totalnum = int(((hight/patchwidth)*2-1)*((width/patchwidth)*2-1))
    xt = np.zeros((totalnum*24,patchwidth,patchwidth,1))
    yt = np.zeros((totalnum*24,patchwidth,patchwidth,1))
    tecount=0
    for ii in range(len(tifdir)):
        four = scipy.io.loadmat(oneaxis + '/' + tifdir[ii])
        g = scipy.io.loadmat(twelve + '/' + tifdir2[ii])
        
        b = four['cutimg'][188:1724,192:2240]
        b1 = g['cutimg'][188:1724,192:2240]
        print('Testing data number:' + str(ii+1),end='\r')
        patch1 = extract_patches(b,(patchwidth,patchwidth),overlap_allowed=0.5)
        patch2 = extract_patches(b1,(patchwidth,patchwidth),overlap_allowed=0.5)

        xt[(tecount)*totalnum:(tecount+1)*totalnum,:,:,0] = patch1
        yt[(tecount)*totalnum:(tecount+1)*totalnum,:,:,0] = patch2
        tecount+=1
    return xt,yt

def unpatchify(patches, imsize):

    assert len(patches.shape) == 4

    i_h, i_w = imsize
    image = np.zeros(imsize, dtype=patches.dtype)
    divisor = np.zeros(imsize, dtype=patches.dtype)

    n_h, n_w, p_h, p_w = patches.shape

    # Calculat the overlap size in each axis
    o_w = (n_w * p_w - i_w) / (n_w - 1)
    o_h = (n_h * p_h - i_h) / (n_h - 1)

    # The overlap should be integer, otherwise the patches are unable to reconstruct into a image with given shape
    assert int(o_w) == o_w
    assert int(o_h) == o_h

    o_w = int(o_w)
    o_h = int(o_h)

    s_w = p_w - o_w
    s_h = p_h - o_h

    for i, j in product(range(n_h), range(n_w)):
        patch = patches[i,j]
        image[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += patch
        divisor[(i * s_h):(i * s_h) + p_h, (j * s_w):(j * s_w) + p_w] += 1

    return image / divisor

def unnormalize(img, totalmax, totalmin):
    newimg = img*(totalmax-totalmin) + totalmin
    return newimg

def saveimg(savepath, xt, yt, test_predict, totalmax, totalmin):
    try:
        os.mkdir(savepath)
    except OSError:
        print ("Creation of the directory %s failed" % savepath)
    else:
        print ("Successfully created the directory %s " % savepath)
    itera = 15
    untestx = unnormalize(xt, totalmax, totalmin)
    untesty = unnormalize(yt, totalmax, totalmin)
    untestp = unnormalize(test_predict, totalmax, totalmin)
    patches = np.reshape(untesty[itera*35:(itera+1)*35,:,:,0],(5,7,512,512))
    reconstructed_imagey = unpatchify(patches, (1536,2048))
    dictforsave = {'image':reconstructed_imagey}
    scipy.io.savemat(savepath + '/1030_label'+str(itera+1)+'.mat',dictforsave)
    
    patches = np.reshape(untestx[itera*35:(itera+1)*35,:,:,0],(5,7,512,512))
    reconstructed_imagex = unpatchify(patches, (1536,2048))
    dictforsave = {'image':reconstructed_imagex}
    scipy.io.savemat(savepath + '/1030_input'+str(itera+1)+'.mat',dictforsave)
    
    patches = np.reshape(untestp[itera*35:(itera+1)*35,:,:,0],(5,7,512,512))
    reconstructed_imagep = unpatchify(patches, (1536,2048))
    dictforsave = {'image':reconstructed_imagep}
    scipy.io.savemat(savepath + '/1030_predict'+str(itera+1)+'.mat',dictforsave)

    minxpy = min(reconstructed_imagey.min(),reconstructed_imagex.min(), reconstructed_imagep.min())
    maxxpy = max(reconstructed_imagey.max(),reconstructed_imagex.max(), reconstructed_imagep.max())
    imsave(savepath + '/groundtruth.jpg',reconstructed_imagey,vmin=minxpy,vmax=maxxpy,cmap='gray')
    imsave(savepath + '/input.jpg',reconstructed_imagex,vmin=minxpy,vmax=maxxpy,cmap='gray')
    imsave(savepath + '/prediction.jpg',reconstructed_imagep,vmin=minxpy,vmax=maxxpy,cmap='gray')

    

if __name__=='__main__':
    #############################################################################
    # load model weights
    script_dir = os.path.dirname(__file__)
    model = unet(pretrained_weights = script_dir + '/DLbasedIDPC_weights.hdf5')
    inputdict = script_dir + '/test_input'
    gtdict = script_dir + '/test_groundtruth'

    #############################################################################
    # load testing datasets
    hight = 1536
    width = 2048
    patchwidth = 512
    xt,yt = idpcimage(inputdict, gtdict, hight, width, patchwidth)
    totalmax = 5.8022953586368935
    totalmin = -2.744483045265948
    xt,yt = normalizeidpc(xt,yt,totalmax,totalmin)
    print('Dataset loading process has done')
    
    #############################################################################
    # predict testing results
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    test_predict = model.predict(xt)
    
    #############################################################################
    # save the predicted images
    savepath = script_dir + "/result_img"
    saveimg(savepath, xt, yt, test_predict, totalmax, totalmin)
