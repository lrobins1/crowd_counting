# Crowd Counting
Crowd_counting is a python library to train and use your own crowd counting models.

## 1) Training dataset 

Choose a already existing dataset ([ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset),[UCF-CC-50](https://www.crcv.ucf.edu/data/ucf-cc-50/),[UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/) or [others](https://paperswithcode.com/datasets?task=crowd-counting)) or annotate your own crowd pictures using matlab.
Use some built-in data augmentation funtion to get better results.

## 2) Groundtruth 

Create the GroundTruth based on the training image.

## 3) Train the model

Train models based on your dataset.

## 4) Predictions

Run the models on new images to have a reliable idea of the number of people on it. <a href="https://github.com/lrobins1/crowd_counting/blob/main/exemples/Ground%20Truth%20generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
