# Crowd Counting
Crowd_counting is a library to train and use your own crowd counting models using python3 and based on tensorflow.
<p align="center">
<img src="https://github.com/lrobins1/crowd_counting/assets/counting_exemple.png" height="128px">
</p>


## 1) Choosing/creating the dataset 

Choose a already existing dataset ([ShanghaiTech](https://github.com/desenzhou/ShanghaiTechDataset),[UCF-CC-50](https://www.crcv.ucf.edu/data/ucf-cc-50/),[UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/), [others](https://paperswithcode.com/datasets?task=crowd-counting)) or annotate your own crowd pictures using matlab.

Use some of our built-in data augmentation function to get better results.

## 2) Groundtruth 

Create the GroundTruth based on the annotated training images.  <a href="https://colab.research.google.com/github/lrobins1/crowd_counting/blob/main/exemples/Ground%20Truth%20generation.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 3) Train the model


Train models based on your dataset directly in python : <a href="https://colab.research.google.com/github/lrobins1/crowd_counting/blob/main/exemples/Model_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

Or directly via the terminal : <a href="https://colab.research.google.com/github/lrobins1/crowd_counting/blob/main/exemples/Model_Training.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

## 4) Predictions

Run the models on new images to have a reliable density map and the estimation of the number of people on it. <a href="https://colab.research.google.com/github/lrobins1/crowd_counting/blob/main/exemples/Model_testing.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


## Bonus : Export your model to json using [tensorflowjs](https://github.com/tensorflow/tfjs/tree/master/tfjs-converter)  to use is in https://our_website.com for crowd counting 

<a href="https://colab.research.google.com/github/lrobins1/crowd_counting/blob/main/exemples/Conversion_to_JS.ipynb"