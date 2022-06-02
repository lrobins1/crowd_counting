"""
This module contain all methods to create and use a CSRNet model for crowd counting
"""

import torch.nn as nn
import torch
from torchvision import models
from .utils import save_net,load_net
import torch.nn.functional as F
import torchvision
import PIL.Image as Image
from matplotlib import pyplot as plt
from .image import *
from matplotlib import cm as CM
from torchvision import datasets, transforms
import pkg_resources
import io

class CSRNet(nn.Module):
    """This class represents a CSRNet model"""
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = _make_layers(self.frontend_feat)
        self.backend = _make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            vgg = models.vgg16(pretrained = True)
            vgg_dict= vgg.state_dict()
            front_dict = self.frontend.state_dict()
            #self._initialize_weights()
            model_dict = []
            for key in front_dict:
               model_dict.append("features."+ key)
            # 1. filter out unnecessary keys
            vgg_dict = {k: v for k, v in vgg_dict.items() if k in model_dict}
            
            vgg_dict_update = {}
            for key in vgg_dict.keys():
                splits = key.split(".")
                newkey = splits[1] + "." + splits[2]
                vgg_dict_update[newkey] = vgg_dict[key] 
                
            # 2. overwrite entries in the existing state dict
            front_dict.update(vgg_dict_update)
            
            # 3. load the new state dict
            self.frontend.load_state_dict(front_dict)
        
            
    def forward(self,x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
                
def _make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
        if dilation:
            d_rate = 2
        else:
            d_rate = 1
        layers = []
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate,dilation = d_rate)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    

def load_model(model_path, use_gpu = True):
    """
    Load a stored already trained CSRNet model.

    :param model_path: the path to the stored model
    :param use_gpu: if the model is loaded on gpu or cpu 

    :returns: the loaded model
    :rtype: CSRNet
    """

    transform=transforms.Compose([
                            transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225]),
                        ])
    model = CSRNet()
    if use_gpu:
      model = model.cuda()
    else:
       model = model.cpu()
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['state_dict'])
    return model
 
def load_pretrained(model_name = 'shanghaiA'):
    """
    Load one of the already pretrained models of the librairy
    
    
    :param model_name: name of the pretrained model to be charged. Can be shanghaiA, shanghaiB, A10
    :returns: the loaded model
    :rtype: CSRNet
    """
    import requests
    #actual_name = 'models/' + model_name + '.pth.tar'
    #model_path = pkg_resources.resource_filename('Crowd_counting', actual_name)
    URL = "https://tfecounting.000webhostapp.com/" + model_name + ".pth.tar"
    print(model_name)
    model_path = requests.get(URL)
    stream = io.BytesIO(model_path.content)
    return load_model(stream)
    

def predict(model,image_path, use_gpu = True):
    """
    Predict the stimated number of people and density map from an image

    :param model: a CSRNet already trained model
    :param image_path: path to the image to be predicted
    :param use_gpu: if the model is loaded on gpu or cpu 

    :returns: the estimated number of people
    :returns: the estimated density map
    """


    from torchvision import datasets, transforms
    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                    ])
    img = transform(Image.open(image_path).convert('RGB')).cuda()
    output = model(img.unsqueeze(0))
    people_nbr = int(output.detach().cpu().sum().numpy())
    return people_nbr, output 
  
def visualize(image, ground_truth = None, model = None, figsize = (100,100)):
    """
    Visualize an image and eventually its ground_truth and one model prediction using matplotlib

    :param image: the path to base image
    :param ground_truth: a .h5 file corresponding to the ground_truth of the image
    :param model: a model used to make a prediction on the base image
    """

  
    if isinstance(ground_truth, str):
      gt_file = h5py.File(ground_truth)
    else: 
      gt_file = ground_truth

    count = 1 #number of things to plot :
    if ground_truth is not None:
      count+=1
      gt = np.asarray(gt_file['density'])
    if model is not None:
      count+=1
      people_nbr, output = predict(model,image)

    plt.figure(figsize = figsize)
    plt.subplot(1,count,1)
    plt.axis('off')
    plt.imshow(Image.open(image))
    plt.title("Base Image", fontsize=75)

    if ground_truth is not None: 
      plt.subplot(1,count,2)
      plt.axis('off')
      plt.imshow(gt,cmap=CM.jet)
      plt.title("Groundtruth : " + str(int(np.sum(gt))), fontsize=75)
  
    if model is not None:
      plt.subplot(1,count,count)
      plt.axis('off')
      plt.imshow(np.squeeze(output.detach().cpu().numpy(),(0,1)),cmap=CM.jet)
      plt.title("Model prediction : " + str(people_nbr), fontsize=75)
    
    plt.show()
     
def evaluate(model,img_paths, MAE=True, MSE=True):
    """
    Compute the MSE-MAE of the model on a designated set of images 

    :param model: a CSRNet already trained model
    :param list image_paths: paths to the images used for evaluation
    :param bool MAE: If set to False the MAE is not computed
    :param bool MSE: If set to False the MSE is not computed


    :returns: the MAE of the model (0 if the MAE parameter is set to False)
    :returns: the MSE of the model (0 if the MSE parameter is set to False)
    """


    mae = 0
    mse = 0
    length = len(img_paths)
    transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                    ])
    for i in range(length):
      img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
      gt_file = h5py.File(img_paths[i].replace('.png','.h5').replace('.jpg','.h5').replace('images','ground_truth'),'r')
      groundtruth = np.asarray(gt_file['density'])
      output = model(img.unsqueeze(0))
      error = output.detach().cpu().sum().numpy()-np.sum(groundtruth)
      if MAE:
        mae += abs(error)
      if MSE:
        mse += (error)**2
      
    return mae/length, mse/length