import pandas as pd
from numpy import *
import numpy as np
from pylab import *
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from skimage.transform import resize
import streamlit as st
import glob
import os

## -- Define and load some usefull info
img_format = [' ','png','jpeg','jpg','tiff']
train_csv = pd.read_csv('/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/train.csv', sep=',') 

plot_folder = '/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/plot/'

current_dir = '/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/sartorius-cell-instance-segmentation/'

## -- Functions for model:
def dice_coef(y_true, y_pred):
    smooth = 1
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

model_dir = '/Users/mitochondria/Documents/Codingschool/DataScienceCourse/Final_Project/Models/'
model = tf.keras.models.load_model(model_dir+'/model_2.keras', 
                                   custom_objects={'dice_coef_loss':dice_coef_loss,'dice_coef':dice_coef})


## -- Functions for building the app
def getSubDir(current_dir):
    sub_dir = [" "]
    tmp_dir =  glob.glob(current_dir+"*", recursive = True)
    for t_dir in tmp_dir:
        sub_dir.append(t_dir.split('/')[-1])
    return sub_dir

def getImgList(img_dir, img_format):
    img_list = [" "]
    file_list = glob.glob(current_dir+"/"+img_dir+"/*."+img_format)
    for file in file_list:
        img_list.append(file.split('/')[-1])
    return img_list


# def histeq(im,nbr_bins=256):
#   """  Histogram equalization of a grayscale image. """

#   # get image histogram
#   imhist,bins = histogram(im.flatten(),nbr_bins,normed=True)
#   cdf = imhist.cumsum() # cumulative distribution function
#   cdf = 255 * cdf / cdf[-1] # normalize

#   # use linear interpolation of cdf to find new pixel values
#   im2 = interp(im.flatten(),bins[:-1],cdf)

#   return im2.reshape(im.shape)

## -- Function for image loading, mask production and image display

def rle_decode(mask_rle, shape):
    '''
    Input [string]: Run-length encoded pixel mask
    Output [tf.array of shape shape]: Segmentation mask
    '''
    shape = tf.convert_to_tensor(shape, tf.int64)
    size = tf.math.reduce_prod(shape)
    # Split string
    s = tf.strings.split(mask_rle)
    s = tf.strings.to_number(s, tf.int64)
    # Get starts and lengths
    starts = s[::2] - 1
    lens = s[1::2]
    # Make ones to be scattered
    total_ones = tf.reduce_sum(lens)
    ones = tf.ones([total_ones], tf.uint8)
    # Make scattering indices
    r = tf.range(total_ones)
    lens_cum = tf.math.cumsum(lens)
    s = tf.searchsorted(lens_cum, r, 'right')
    idx = r + tf.gather(starts - tf.pad(lens_cum[:-1], [(1, 0)]), s)
    # Scatter ones into flattened mask
    mask_flat = tf.scatter_nd(tf.expand_dims(idx, 1), ones, [size])
    # Reshape into mask
    return tf.reshape(mask_flat, shape)


def createMask(img_id, new_shape):
#     parent_dir = '/kaggle/working/'
#     mask_dir = os.path.join(parent_dir, 'masks')
#     if os.path.exists(mask_dir):
#         shutil.rmtree(mask_dir)
#     os.makedirs(mask_dir)
        
    shape = [train_csv.loc[train_csv.id == img_id, 'height'].tolist()[0], train_csv.loc[train_csv.id == img_id, 'width'].tolist()[0]]
    mask = np.zeros(shape).astype(float)
    buffer = 0. * mask
    for idx, row in train_csv[train_csv.id == img_id].iterrows():
        # decode the mask
        buffer = rle_decode(row.annotation, shape).numpy().astype(float)
        mask += buffer
    mask[mask > 0] = 1
    mask[mask == 0] = np.nan
    mask = tf.reshape(mask, [new_shape]).numpy()
    return mask
        
def img_color_rescale(img):
    return(img-tf.reduce_mean(img)/128)


def loadImg(id_img, img_format, img_dir, img_type):
    img_path = current_dir+img_dir+'/'+id_img+'.'+img_format
    if img_type == 'org':
        img = tf.keras.utils.load_img(img_path, color_mode = 'grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = resize(img, (512, 512, 1), mode='constant', preserve_range=True)
        #img = tf.reshape(img, [512,512]).numpy()
        return img
    elif img_type == 'true':
        mask = createMask(id_img, [512,512])
        return mask
    else:
        img = tf.keras.utils.load_img(img_path, color_mode = 'grayscale')
        img = tf.keras.preprocessing.image.img_to_array(img)
        img = resize(img, (512, 512, 1), mode='constant', preserve_range=True)
        img.append(img_color_rescale(img).numpy())
        img = np.array(img)
        pred_mask = model.predict(img[0].reshape((1,512,512,1)))[0]
        pred_mask = tf.reshape(pred_mask, [512,512]).numpy()
        return pred_mask


def getFig(img_path):
    img = Image.open(img_path)
    img = np.array(img)
    
    fig = plt.figure(figsize=(16,16), clear=True)
    ax = plt.gca()
    #img = img_color_rescale(np.array(Image.open(img_filename)))
    ax.imshow(img, zorder=-3, cmap = 'gray', rasterized=True)
    shape = img.shape
    del img
    # rasterize to conserve RAM
    ax.set_rasterization_zorder(0)
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)
        
    return fig

def getFigMask(current_dir, sub_dir, id_img, img_format):
    img = Image.open(current_dir+sub_dir+'/'+id_img+'.'+img_format)
    img = np.array(img)
    
    fig = plt.figure(figsize=(16,16), clear=True)
    ax = plt.gca()
    
    # plot the image itself
    ax.imshow(img, zorder=-3, cmap = 'gray', rasterized=True)
    shape = img.shape
    del img
    
    # overplot segmentation
    mask = np.zeros(shape).astype(float)
    buffer = 0. * mask
    for idx, row in train_csv[train_csv.id == id_img].iterrows():
        # decode the mask
        buffer = rle_decode(row.annotation, shape).numpy().astype(float)
        mask += buffer

    # draw the mask
    mask[mask > 0] = 1
    mask[mask == 0] = np.nan
    cmap = cm.get_cmap('YlGn_r').copy()
    cmap.set_bad(alpha=0.0)
    ax.imshow(mask, cmap=cmap, alpha=0.3, zorder=-2, rasterized=True)
    del mask, buffer
    
    # rasterize to conserve RAM
    ax.set_rasterization_zorder(0)
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)
        
    return fig

def getFigPred(current_dir, sub_dir, id_img, img_format):
    img = Image.open(current_dir+sub_dir+'/'+id_img+'.'+img_format)
    img = np.array(img)
    
    fig = plt.figure(figsize=(16,16), clear=True)
    ax = plt.gca()
    
    # plot the image itself
    ax.imshow(img, zorder=-3, cmap = 'gray', rasterized=True)
    shape = img.shape
    del img
    
    # get the prediction mask
    img_model = tf.keras.utils.load_img(current_dir+sub_dir+'/'+id_img+'.'+img_format, color_mode = 'grayscale')
    img_model = tf.keras.preprocessing.image.img_to_array(img_model)
    img_model = resize(img_model, (512, 512, 1), mode='constant', preserve_range=True)
    img_model = img_color_rescale(img_model).numpy()
    
    pred_mask = model.predict(img_model.reshape((1,512,512,1)))[0]
    pred_mask = resize(pred_mask, (shape), mode='constant', preserve_range=True)
    pred_mask = pred_mask.reshape(shape)
    mask_shape = pred_mask.shape
    
    
    # draw the mask
    pred_mask[pred_mask > 0.65] = 1
    pred_mask[pred_mask <= 0.65] = np.nan
    cmap = cm.get_cmap('YlGn_r').copy()
    cmap.set_bad(alpha=0.0)
    ax.imshow(pred_mask, cmap=cmap, alpha=0.3, zorder=-2, rasterized=True)
    del pred_mask
    
    # rasterize to conserve RAM
    ax.set_rasterization_zorder(0)
    x_axis = ax.axes.get_xaxis()
    x_axis.set_visible(False)
    
    y_axis = ax.axes.get_yaxis()
    y_axis.set_visible(False)
        
    return fig


def rle_encode(mask):
    """
    mask: numpy array with 1 = mask and 0 = background
    Returns rle as string formatted
    """
    pixels = mask.flatten()
    pixels = np.concatenate([[0], pixels,[0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0]+1
    runs[1::2] -=runs[::2]
    return ' '.join(str(x) for x in runs)


def getMask(current_dir, sub_dir, id_img, img_format):
    shape = np.array(Image.open(current_dir+sub_dir+'/'+id_img+'.'+img_format)).shape
    img_model = tf.keras.utils.load_img(current_dir+sub_dir+'/'+id_img+'.'+img_format, color_mode = 'grayscale')
    img_model = tf.keras.preprocessing.image.img_to_array(img_model)
    img_model = resize(img_model, (512, 512, 1), mode='constant', preserve_range=True)
    img_model = img_color_rescale(img_model).numpy()
    
    pred_mask = model.predict(img_model.reshape((1,512,512,1)))[0]
    pred_mask = resize(pred_mask, (shape), mode='constant', preserve_range=True)
    pred_mask = pred_mask.reshape(shape)
    
    # draw the mask
    pred_mask[pred_mask > 0.65] = 1
    pred_mask[pred_mask <= 0.65] = np.nan
    return np.array(pred_mask)

def getDataFrame(current_dir, sub_dir, id_img, img_format): 
    width, height=(Image.open(current_dir+sub_dir+'/'+id_img+'.'+img_format)).size
    cell_type = sub_dir
    mask = getMask(current_dir, sub_dir, id_img, img_format)
    annotation = rle_encode(mask)
    tmp_df = pd.DataFrame({
        'id':[id_img],
        'annotation':[annotation],
        'width':[width],
        'height':[height],
        'cell_type':[cell_type]
    })
    return tmp_df