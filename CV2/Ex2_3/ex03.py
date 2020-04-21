from __future__ import print_function
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.max_pool2d(F.relu(self.conv1(xb)), 2)
        xb = F.max_pool2d(F.relu(self.conv2(xb)), 2)
        xb = xb.view(-1, 32*7*7)
        xb = F.relu(self.fc1(xb))
        xb = self.fc2(xb)
        return xb
        

def read_file(filename):
	lines = []
	with open(filename, 'r') as file:
	    for line in file: 
	        line = line.strip() #or some other preprocessing
	        lines.append(line)
	return lines


class FixationDataset(Dataset):
	def __init__(self, root_dir, image_file, fixation_file, transform=None):
		self.root_dir = root_dir
		self.image_files = read_file(os.path.join(root_dir, image_file))
		self.fixation_files = read_file(os.path.join(root_dir, fixation_file))
		self.transform = transform
		assert(len(self.image_files) == len(self.fixation_files))

	def __len__(self):
		return len(self.image_files)

	def __getitem__(self, idx):
		if torch.is_tensor(idx):
			idx = idx.tolist()

		img_name = os.path.join(self.root_dir, self.image_files[idx])
		image = imageio.imread(img_name)

		fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
		fix = imageio.imread(fix_name)

		sample = {'image': image, 'fixation': fix}

		if self.transform:
			sample = self.transform(sample)

		return sample
		
class Rescale():
    def __init__(self):
        pass
        
    def __call__(self, sample):
        for key in sample:
            sample[key] = sample[key].astype(float) / 255.0
        return sample

class ToTensor():
    def __init__(self):
        pass
        
    def __call__(self, sample):
        for key in sample:
            content = torch.from_numpy(sample[key])
            shp = content.shape
            if content.ndim == 2:
                content = content.unsqueeze(-1)
            # Rearrange channels to first dim:
            content = content.permute(2, 0, 1)
            # Store it:
            sample[key] = content
        return sample

def save(epoch, model, opt, loss, path):
    torch.save({"epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss
                }, path)
            
normalize = transforms.Normalize(mean=[0.485 , 0.456 , 0.406], std =[0.229 , 0.224 , 0.225])
# TODO: add to GPU transform
composed = torchvision.transforms.Compose([ToTensor(), normalize])
# TODO: add data augmentation transforms: https://pytorch.org/docs/stable/torchvision/transforms.html
ds = FixationDataset("data", "train_images.txt", "train_fixations.txt", composed)        
sampler = RandomSampler(ds)
dl = DataLoader(ds, batch_size=16, num_workers=2, pin_memory=0, sampler=sampler)

import torchvision.models as models
vgg16 = models.vgg16(pretrained=True)
for param in vgg16.parameters():
    param.requires_grad = False
num_feats = vgg16.fc.in_features
num_out_classes = 3
vgg16.fc = torch.nn.Linear(num_feats, num_out_classes)
opt = torch.optim.Adam(model.parameters(), lr=0.0003)

epochs = 5
pbar = tqdm(total=epochs)
best_loss = None
for epoch in range(epochs):
    pbar.update(1)
    total_train_loss = 0
    model.train()
    for batch in dl:
        img = batch["image"]
        fixation = batch["fixation"]
        
        pred = model(img)
        loss = loss_func(pred, fixation)
        
        loss.backward()
        opt.step()
        opt.zero_grad()
        
        total_train_loss += loss.detach()
        
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_func(model(val_img), val_fixation) 
                       for val_img, val_fixation in val_dl) / len(val_dl)
    
    pbar.set_postfix("loss": total_train_loss, "val_loss", val_loss)
    save(epoch, model, opt, loss, "newest_model.pt")
    if best_loss is None or val_loss <= best_loss):
        best_loss = val_loss
        save(epoch, model, opt, loss, "best_model.pt")
   
pbar.close()
      
            
            
            
quit()
		
		
xtrain = torch.from_numpy(np.load("x_train.npy"))
ytrain = torch.from_numpy(np.load("y_train.npy"))
from torch.utils.data import TensorDataset
trainds = TensorDataset(xtrain , ytrain)
print("Size of dataset: ", len(trainds))
# Get n ’ th  sample
n = 1000
x, y = trainds[n]
# F i r s t  16  samples
xb, yb = trainds[:16]
print(xb.shape, yb.shape)
# I t e r a t i n g  over  e n t i r e  d a t a s e t
#for x, y in trainds:
#    print(x.shape, y.shape)
#    print(y)
#    # do  something  with  sample  x ,  ypass

net = MNIST_CNN()
loss_fcn = F.cross_entropy
opt = torch.optim.SGD(net.parameters(), lr=0.1)

from torch.utils.data import DataLoader
traindl = DataLoader(trainds , batch_size=16)
epochs = 10
for epoch in range(epochs):
    
    for xb, yb in traindl:
        pbar.update(1)
        pred = net(xb)
        loss = loss_fcn(pred, yb)
        print(loss)


        loss.backward()
        opt.step()
        opt.zero_grad()
    pbar.close()




