import os

from tqdm import tqdm
import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torchvision
from apex import amp
#import torchvision.models as models


class MNIST_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 7 * 7, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.max_pool2d(F.ReLU(self.conv1(xb)), 2)
        xb = F.max_pool2d(F.ReLU(self.conv2(xb)), 2)
        xb = xb.view(-1, 32*7*7)
        xb = F.ReLU(self.fc1(xb))
        xb = self.fc2(xb)
        return xb
    
    
class Encoder(nn.Module):
    def __init__(self, pretrained=True):
        super(Encoder, self).__init__()
        
        # TODO: possibly freeze some or all of the weights here
        
        # Get feature extractor of vgg16 but do not use last pooling layer:
        self.feature_extractor = torchvision.models.vgg16(pretrained=pretrained).features[:-1] 
            
    def forward(self, x):
        features = self.feature_extractor(x)
        return features
    
class Decoder(nn.Module):
    def __init__(self, input_shp=(512, 14)):
        super().__init__()
        
        in_channels, size = input_shp 
        
        #TODO: try upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1 instead of Upsample
        
        block1 = nn.Sequential(nn.Conv2d(in_channels, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1),
                               nn.ReLU(),
                               nn.Upsample(scale_factor=2, mode="bilinear")
                              )
        block2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1),
                               nn.ReLU(),
                               nn.Upsample(scale_factor=2, mode="bilinear")
                              )
        block3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(256, 256, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(256, 256, 3, padding=1),
                               nn.ReLU(),
                               nn.Upsample(scale_factor=2, mode="bilinear")
                              )
        block4 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(128, 128, 3, padding=1), 
                               nn.ReLU(),
                               nn.Upsample(scale_factor=2, mode="bilinear")
                              )
        block5 = nn.Sequential(nn.Conv2d(128, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 1, 3, padding=1), 
                               #nn.Sigmoid()
                              )
        self.generator = nn.Sequential(block1, block2, block3, block4, block5)
    
    def forward(self, x):
        return self.generator(x)

class EncoderDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder(pretrained=True)
        self.decoder = Decoder((512, 14))
        
    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
    
class BCELossDownsampling():
    def __init__(self):
        self.downsample = torch.nn.AvgPool2d(4, stride=4, count_include_pad=False)
        self.loss_fcn = torch.nn.BCEWithLogitsLoss()
        
    def __call__(self, pred, target):
        return self.loss_fcn(self.downsample(pred), self.downsample(target))

def read_file(filename):
	lines = []
	with open(filename, 'r') as file:
	    for line in file: 
	        line = line.strip() #or some other preprocessing
	        lines.append(line)
	return lines


class FixationDataset(Dataset):
    def __init__(self, root_dir, image_file, fixation_file, input_transform=None, target_transform=None):
        self.root_dir = root_dir
        self.image_files = read_file(os.path.join(root_dir, image_file))
        self.fixation_files = read_file(os.path.join(root_dir, fixation_file))
        self.input_transform = input_transform
        self.target_transform = target_transform
        assert(len(self.image_files) == len(self.fixation_files))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)
        if self.input_transform:
            image = self.input_transform(image)

        fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
        fix = imageio.imread(fix_name)
        if self.input_transform:
            fix = self.target_transform(fix)

        sample = {'image': image, 'fixation': fix}

        #if self.transform:
        #	sample = self.transform(sample)
        return sample
		
class Rescale():
    def __init__(self):
        pass
    def __call__(self, sample):
        return sample.astype(float) / 255.0
        for key in sample:
            sample[key] = sample[key].astype(float) / 255.0
        return sample

class ToTensor():
    def __init__(self):
        pass
        
    def __call__(self, sample):
        content = torch.from_numpy(sample).float()
        #print(content.shape)
        shp = content.shape
        if content.ndim == 2:
            content = content.unsqueeze(-1)
        # Rearrange channels to first dim:
        content = content.permute(2, 0, 1)
        #print(content.shape)
        return content
        
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
            
means = torch.tensor([0.485 , 0.456 , 0.406])
stds = torch.tensor([0.229 , 0.224 , 0.225])
normalize = torchvision.transforms.Normalize(mean=means, std=stds)
input_transform = torchvision.transforms.Compose([Rescale(), ToTensor(), normalize])
target_transform = torchvision.transforms.Compose([Rescale(), ToTensor()])

# TODO: add data augmentation transforms: https://pytorch.org/docs/stable/torchvision/transforms.html
ds = FixationDataset("data", "train_images.txt", "train_fixations.txt", input_transform, target_transform)      
val_ds = FixationDataset("data", "val_images.txt", "val_fixations.txt", input_transform, target_transform)        

sampler = RandomSampler(ds)
torch.backends.cudnn.benchmark = True
pin_mem = False
train_batch_size = 16
val_batch_size = 9
train_dl = DataLoader(ds, batch_size=train_batch_size, num_workers=2, pin_memory=pin_mem, sampler=sampler)
val_dl = DataLoader(val_ds, batch_size=val_batch_size, num_workers=2, pin_memory=pin_mem)

#vgg16 = models.vgg16(pretrained=True)
#print(vgg16)
#for param in vgg16.parameters():
#    param.requires_grad = False
#num_feats = vgg16.fc.in_features
#num_out_classes = 3
#vgg16.fc = torch.nn.Linear(num_feats, num_out_classes)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EncoderDecoder()
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=0.0003)
loss_fcn = BCELossDownsampling()
os.makedirs("imgs", exist_ok=True)

use_apex = True
if use_apex:
    model, optimizer = amp.initialize(model, opt, opt_level="O1")

epochs = 20
pbar = tqdm(total=epochs * len(train_dl))
best_loss = None
for epoch in range(epochs):
    
    model.train()
    total_train_loss = 0
    for batch in train_dl:
        pbar.update(1)
        img, fixation = batch["image"].to(device), batch["fixation"].to(device)
        
        pred = model(img)
        loss = loss_fcn(pred, fixation)
        
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        opt.step()
        opt.zero_grad()
        
        total_train_loss += loss.detach().item()
                
    
    model.eval()
    with torch.no_grad():
        val_loss = sum(loss_fcn(model(val_sample["image"].to(device)), val_sample["fixation"].to(device)) 
                       for val_sample in val_dl) / len(val_dl) * val_batch_size
        val_loss = val_loss.detach().item()
        
        # Save images:
        def stack_img_pred_and_fix(sample):
            use_means = means.view(1, 3, 1, 1)
            use_stds = stds.view(1, 3, 1, 1)
            img = sample["image"].mul_(use_stds).add_(use_means) #* 255
            #img = ((sample["image"] * stds) + means) * 255
            pred = model(img.to(device)).cpu()
            # turn to 3 channel:
            pred = torch.cat([pred] * 3, dim=1)
            fix = sample["fixation"]
            fix = torch.cat([fix] * 3, dim=1)
            
            
            #img = sample["image"]
            #pred = model(img.to(device)).cpu()
            gray_img = img.mean(dim=1).unsqueeze(1)
            #fix = sample["fixation"]
            
            #print(img.shape, pred.shape, gray_img.shape, fix.shape)
            #return torch.cat([gray_img, pred, fix], dim=1)
            shp = pred.shape
            mod_shp = list(shp)
            mod_shp[0] *= 3
            stacked = torch.stack([img, pred, fix], dim=1)
            #print(stacked.shape)
            return stacked.view(mod_shp)
        imgs = torch.cat([stack_img_pred_and_fix(sample) for sample in val_dl])
        torchvision.utils.save_image(imgs[:100], "imgs/" + str(epoch) + "_val_preds.png", nrow=9)
    
    #pbar.set_postfix({"loss": total_train_loss / len(train_dl), "val_loss": val_loss})
    print("loss: ", round(total_train_loss / len(train_dl) * train_batch_size, 2), " val_loss: ", round(val_loss, 2))
    save(epoch, model, opt, loss, "newest_model.pt")
    if best_loss is None or val_loss <= best_loss:
        best_loss = val_loss
        save(epoch, model, opt, loss, "best_model.pt")
        
    
   
pbar.close()

# Save images:
with torch.no_grad():
    imgs = torch.cat([model(sample["image"].to(device)) for sample in val_dl][:-1])
    torchvision.utils.save_image(imgs[:100], "imgs/final_val_preds.png", nrow=10)
      
            
            
            
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




