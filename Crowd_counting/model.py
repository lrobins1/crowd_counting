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

import pkg_resources

class CSRNet(nn.Module):
    def __init__(self, load_weights=True):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512]
        self.backend_feat  = [512, 512, 512,256,128,64]
        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat,in_channels = 512,dilation = True)
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)
        if not load_weights:
            mod = models.vgg16(pretrained = True)
            self._initialize_weights()
            #for i in range(len(self.frontend.state_dict().items())):
                #self.frontend.state_dict().items()[i][1].data[:] = mod.state_dict().items()[i][1].data[:]
            self.frontend.state_dict = mod.state_dict().copy()
    def forward(self,x):
        '''
        x_len = x.shape[2] 
        y_len = x.shape[3]
        scale_factor = 1
        if x_len or y_len > 1024:
            if x_len > y_len:
                scale_factor = 1024/x_len
            else:
                scale_factor = 1024/y_len
            x = F.interpolate(x, scale_factor=(scale_factor, scale_factor))
            x = torchvision.transforms.functional.normalize(x,mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        '''
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
            
def make_layers(cfg, in_channels = 3,batch_norm=False,dilation = False):
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
    
#return the loaded model located at model_path (or the basic shangai partAmodel if no path is given ) 
#using gpu or using cpu (if the use_gpu parameter is set to False)
def load_model(model_path, use_gpu = True):
  from torchvision import datasets, transforms
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
 
def load_best():
    model_path = pkg_resources.resource_filename('Crowd_counting', 'data/PartAmodel_best.pth.tar')
    print(model_path)
    return load_model(model_path)
    
  
#To plot the image density map from the output, use : 
#plt.imshow(np.squeeze(output.detach().cpu().numpy(),(0,1)),cmap=CM.jet)
def predict(model,image_path, use_gpu = True):
  from torchvision import datasets, transforms
  transform=transforms.Compose([
                        transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
                    ])
  img = transform(Image.open(image_path).convert('RGB')).cuda()
  output = model(img.unsqueeze(0))
  people_nbr = int(output.detach().cpu().sum().numpy())
  return people_nbr, output 
  
#amelioration : take lists 
#imag = PIL image or path to image
def visualize(image, ground_truth = None, model = None, figsize = (100,100)):

  if isinstance(image, str):
    image = Image.open(image)
  
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
  plt.imshow(image)
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
    
#calculate and print the MAE of the models
def eval(model,img_paths, MAE=True, MSE=True):
  mae = 0
  mse = 0
  length = len(img_paths)
  for i in range(length):
      img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
      gt_file = h5py.File(img_paths[i].replace('.png','.h5').replace('images','ground_truth'),'r')
      groundtruth = np.asarray(gt_file['density'])
      output = model(img.unsqueeze(0))
      error = output.detach().cpu().sum().numpy()-np.sum(groundtruth)
      if MAE:
        mae += abs(error)
      if MSE:
        mse += (error)**2
      
  return mae/length, mse/length