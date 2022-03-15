import h5py
import torch
import shutil

def save_net(fname, net):
    with h5py.File(fname, 'w') as h5f:
        for k, v in net.state_dict().items():
            h5f.create_dataset(k, data=v.cpu().numpy())
def load_net(fname, net):
    with h5py.File(fname, 'r') as h5f:
        for k, v in net.state_dict().items():        
            param = torch.from_numpy(np.asarray(h5f[k]))         
            v.copy_(param)
            
def save_checkpoint(state, is_best,task_id, filename='/content/gdrive/My Drive/TFE crowd counting/CSRNet-pytorch/checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename,'/content/gdrive/My Drive/TFE crowd counting/CSRNet-pytorch/model_best.pth.tar')
        

#return the loaded model located at model_path (or the basic shangai partAmodel if no path is given ) 
#using gpu or using cpu (if the use_gpu parameter is set to False)
def load_model(model_path, use_gpu = False):
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

#To plot the image density map from the output, use : 
#plt.imshow(np.squeeze(output.detach().cpu().numpy(),(0,1)),cmap=CM.jet)
def predict(model,image_path, use_gpu = True):
  img = transform(Image.open(image_path).convert('RGB')).cuda()
  output = model(img.unsqueeze(0))
  people_nbr = int(output.detach().cpu().sum().numpy())
  return people_nbr, output  