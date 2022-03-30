import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import os
import glob
import scipy.io as io
import numpy as np
import random

#améliorations : saving directory + no saving (list return)

'Flip a list of images horizontally and change the .mat file assosiated with it'
def Horizontal_Flip(img_paths,GT_paths,register = True, img_format = 'jpg'):
  #correct arguments
  assert img_format == 'jpg' or img_format == 'png', 'img_format can only take the values : jpg, png'
  
  for img_path,GT_path in list(zip(img_paths, GT_paths)):
      #load
      img = cv2.imread(img_path)
      mat = io.loadmat(GT_path)
      #flip img
      new_im = cv2.flip(img,1)
      #flip coordinates in .mat
      information = mat['image_info']
      i=information[0][0]
      x=i.item(0)
      for j in range(len(x[0])):
        x[0][j][0]= img.shape[1]-x[0][j][0]

      #save image
      if register==True:
        if img_format == 'jpg':
          filename = img_path.replace('.jpg','_reverse.jpg')
        if img_format == 'png':
          filename = img_path.replace('.png','_reverse.png')
        
        matname = GT_path.replace('.mat','_reverse.mat')
        cv2.imwrite(filename, new_im)
        io.savemat(matname, mat)

def brightness_variation(img_paths,GT_paths,register = True, img_format = 'jpg'):
  #correct arguments
  assert img_format == 'jpg' or img_format == 'png', 'img_format can only take the values : jpg, png'

  datagen = ImageDataGenerator(brightness_range=[0,1])
  dic = {}
  for img_path,GT_path in list(zip(img_paths, GT_paths)):
    img = cv2.imread(img_path)
    mat = io.loadmat(GT_path)
    dic['brightness'] = random.uniform(0.3,0.65)
    new_im = datagen.apply_transform(img, dic)

    #save image
    if register == True:
      if img_format == 'jpg':
        filename = img_path.replace('.jpg','_bright.jpg')
      if img_format == 'png':
        filename = img_path.replace('.png','_bright.png')
    matname = GT_path.replace('.mat','_bright.mat')
    cv2.imwrite(filename, new_im)
    io.savemat(matname, mat)

def reverse_and_bright(img_paths,GT_paths,register = True,img_format = 'jpg'):
    #correct arguments
    assert img_format == 'jpg' or img_format == 'png', 'img_format can only take the values : jpg, png'
    for img_path,GT_path in list(zip(img_paths, GT_paths)):
      img = cv2.imread(img_path)
      mat = io.loadmat(GT_path)
      datagen = ImageDataGenerator(brightness_range=[0,1])
      dic = {}
      new_image = cv2.flip(img,1)
      dic['brightness'] = random.uniform(0.3,0.65)
      new_im = datagen.apply_transform(new_image, dic)
      #flip coordinates in .mat
      information = mat['image_info']
      i=information[0][0]
      x=i.item(0)
      for j in range(len(x[0])):
        x[0][j][0]= img.shape[1]-x[0][j][0]


      if register==True:
        if img_format == 'jpg':
          filename = img_path.replace('.jpg','_combine.jpg')
        if img_format == 'png':
          filename = img_path.replace('.png','_combine.png')
        
        matname = GT_path.replace('.mat','_combine.mat')
        cv2.imwrite(filename, new_im)
        io.savemat(matname, mat)

def full_augment(img_paths,GT_paths, img_format = 'jpg'):
  Horizontal_Flip(img_paths,GT_paths,img_format = img_format)
  brightness_variation(img_paths,GT_paths,img_format = img_format)
  reverse_and_bright(img_paths,GT_paths,img_format = img_format)


def dir_to_list(img_dir,img_format ='jpg'):
    img_paths = []
    GT_paths = []
    
    #img into list
    if img_format == 'jpg':
      for img_path in glob.glob(os.path.join(img_dir, '*.jpg')):
          img_paths.append(img_path)
          GT_path = img_path.replace('images','ground_truth').replace('.jpg','.mat')
          GT_paths.append(GT_path)
    
    if img_format == 'png':
      for img_path in glob.glob(os.path.join(img_dir, '*.png')):
          img_paths.append(img_path)
          GT_path = img_path.replace('images','ground_truth').replace('.png','.mat')
          GT_paths.append(GT_path)
    
    return img_paths, GT_paths