import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import glob
from scipy.ndimage.filters import gaussian_filter 
import scipy
from scipy import spatial
from image import *

#this is borrowed from https://github.com/davideverona/deep-crowd-counting_crowdnet
#This function create the density by using adaptive kernels
def gaussian_filter_density(gt):
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density
    pts = list(zip(np.nonzero(gt)[1],np.nonzero(gt)[0]))  
    pts = np.array(pts)
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1],pt[0]] = 1.
        if gt_count > 1:
            sigma = (distances[i][1]+distances[i][2]+distances[i][3])*0.1
        else:
            sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
        density += scipy.ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
    return density
    

#Generate h5 ground_truth file based on the paths to the images file as explained in the paper
#Crowded = False -> Usage of only gaussian // Crowded = True -> Usage of geometric adaptive kernel

def gt_gen(img_paths, crowded = True ,Verbose = False):
  count=0
  for img_path in img_paths:
      mat = io.loadmat(img_path.replace('.jpg','.mat').replace('images','ground_truth').replace('IMG_','GT_IMG_'))
      img= plt.imread(img_path)
      k = np.zeros((img.shape[0],img.shape[1]))
      gt = mat["image_info"][0,0][0,0][0]
      for i in range(0,len(gt)):
          if int(gt[i][1])<img.shape[0] and int(gt[i][0])<img.shape[1]:
              k[int(gt[i][1]),int(gt[i][0])]=1.
      
      #density generation for crowded image
      if crowded:
        k = gaussian_filter_density(k)
      #density generation for non crowded image
      else:
        k = gaussian_filter(k,15)


      with h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'), 'w') as hf:
              hf['density'] = k
      if Verbose:
        print("image "+str(count) + "/" + str(len(img_paths))+ " done")
        count+=1
