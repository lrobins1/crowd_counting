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
       

#To plot the image density map from the output, use : 
#plt.imshow(np.squeeze(output.detach().cpu().numpy(),(0,1)),cmap=CM.jet)
def predict(model,image_path, use_gpu = True):
  img = transform(Image.open(image_path).convert('RGB')).cuda()
  output = model(img.unsqueeze(0))
  people_nbr = int(output.detach().cpu().sum().numpy())
  return people_nbr, output  