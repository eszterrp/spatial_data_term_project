#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 12:10:18 2022

@author: pazma
"""
# Modules
import numpy as np
from matplotlib import pyplot
from skimage import color, data, exposure, feature, filters, io, morphology, segmentation, transform, util
from image_utilities import *
import tensorflow as tf

import sklearn
from sklearn.model_selection import train_test_split
import pandas as pd
import PIL
import os
from PIL import Image
import numpy as np
import sklearn
import keras
from PIL import Image
from numpy import asarray
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tensorflow.keras import layers
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import AveragePooling2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt 
import keras.backend as K
from keras.losses import binary_crossentropy
from scipy import ndimage, misc


# Augmentation functions         
def random_horizontal_flip(input_img,gt_img,prob): 
    if np.random.rand()< prob:
        input_img = np.fliplr(input_img)
        gt_img = np.fliplr(gt_img)
    return input_img,gt_img    

def random_vertical_flip(input_img, gt_img,prob=0.5):
    if np.random.rand()< prob:
        input_img = np.flipud(input_img)
        gt_img = np.flipud(gt_img)
    return input_img,gt_img

def random_rotation(input_img, gt_img,prob=0.5): 
    if np.random.rand()< prob: 
        input_img = ndimage.rotate(input_img,90)
        gt_img = ndimage.rotate(gt_img,90)
    return input_img,gt_img 


# Loss functions
import keras.backend as K
from keras.losses import binary_crossentropy

smooth = 1e-15
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = y_true_f * y_pred_f
    score = (2. * K.sum(intersection) + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1. - score

def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

# Evaluation metric
def iou(y_true,y_pred):
    def f(y_true,y_pred):
        intersection = (y_true*y_pred).sum()
        union = y_true.sum() + y_pred.sum() - intersection
        x = (intersection + 1e-15) / (union + 1e-15)
        x = x.astype(np.float32)
        return x
    return tf.numpy_function(f,[y_true,y_pred],tf.float32)


# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map

class land_cover_DataGenerator(tf.keras.utils.Sequence):
    
    def __init__(self, path_to_images,
                 batch_size,
                 input_size=(224, 224, 3),
                 output_size=(224,224,7),
                 shuffle=True, class_rgb_values=None, augmentation=False, augmentation_prob=0.5):
        
        self.batch_size = batch_size
        self.input_size = input_size
        self.class_rgb_values = class_rgb_values
        self.shuffle = shuffle
        self.output_size = output_size
        self.augmentation = augmentation
        self.augmentation_prob = augmentation_prob
        self.input_list = []
        self.gt_list = []
        self.path_to_images = path_to_images
        self.list_path_to_images = os.listdir(path_to_images)
        for i in self.list_path_to_images:
            if 'sat' in i:
                self.input_list.append(i)
            if 'mask' in i:
                self.gt_list.append(i)  
        self.n = len(self.input_list)

    def __getitem__(self, index):
        batch_input_paths = self.input_list[index * self.batch_size:(index + 1) * self.batch_size]
        batch_gt_paths = self.gt_list[index * self.batch_size:(index + 1) * self.batch_size]
        x= np.zeros((self.batch_size,)+ self.input_size)
        y= np.zeros((self.batch_size,)+ self.output_size)
        for i,(input_im,gt_im) in enumerate(zip(batch_input_paths,batch_gt_paths)): 
            img_in = load_img(self.path_to_images+input_im,target_size=self.input_size)
            img_gt = load_img(self.path_to_images+gt_im,target_size=self.input_size)
            
            if self.augmentation: 
                img_in, img_gt = random_horizontal_flip(img_in,img_gt,prob=self.augmentation_prob)
                img_in, img_gt= random_vertical_flip(img_in, img_gt,prob=self.augmentation_prob)
                img_in, img_gt = random_rotation(img_in, img_gt,prob=self.augmentation_prob)
            
            img_gt_encoded = one_hot_encode(img_gt, self.class_rgb_values).astype('float')
            
            x[i] = img_in
            y[i] = img_gt_encoded
        
        x = x.astype('float32')
        y = y.astype('float32')
        x = x/ 255
        return x, y
    
    def __len__(self):
        return self.n // self.batch_size
    
# PREDICTION STATISTICS
def display_statistics(image_test:np.ndarray, label_test:np.ndarray, proba_predict:np.ndarray, label_predict:np.ndarray):
    # Format
    image_test    = (image_test * 255).astype(int)
    label_test    = label_test.astype(bool)
    label_predict = label_predict.astype(bool)
    # Statistics
    mask_tp = np.logical_and(label_test, label_predict)
    mask_tn = np.logical_and(np.invert(label_test), np.invert(label_predict))
    mask_fp = np.logical_and(np.invert(label_test), label_predict)
    mask_fn = np.logical_and(label_test, np.invert(label_predict))
    # Augmented images
    colour  = (255, 255, 0)
    images  = [np.where(np.tile(mask, (1, 1, 3)), colour, image_test) for mask in [mask_tp, mask_tn, mask_fp, mask_fn]]
    # Figure
    images = [image_test, label_test, proba_predict, label_predict] + images
    titles = ['Test image', 'Test label', 'Predicted probability', 'Predicted label', 'True positive', 'True negative', 'False positive', 'False negative']
    fig, axs = pyplot.subplots(2, 4, figsize=(20, 10))
    for image, title, ax in zip(images, titles, axs.ravel()):
        ax.imshow(image)
        ax.set_title(title, fontsize=20)
        ax.axis('off')
    pyplot.tight_layout(pad=2.0)
    pyplot.show()

'''
# Computes prediction statistics
for i in np.random.choice(range(len(images_test)), 5, replace=False):
    display_statistics(image_test=images_test[i], label_test=labels_test[i], proba_predict=probas_predict[i], label_predict=labels_predict[i])
'''