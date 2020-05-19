import os

from tqdm import tqdm
import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torchvision
from torch.utils.tensorboard import SummaryWriter
from apex import amp
#import torchvision.models as models
    
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x
    
class Encoder(nn.Module):
    def __init__(self, net_type, freeze, pretrained=True):
        super(Encoder, self).__init__()
        
        self.net = net_type
        
        if self.net == "res":
            self.feature_extractor = torchvision.models.resnet50(pretrained=pretrained)
            print(self.feature_extractor)
            self.feature_extractor.avgpool = Identity()
            self.feature_extractor.fc = Identity()
        elif self.net == "vgg":
            # Get feature extractor of vgg16 but do not use last pooling layer:
            self.feature_extractor = torchvision.models.vgg16(pretrained=pretrained).features[:-1] 
            
        if freeze == "all":
            # Freeze params:
            for parameter in self.feature_extractor.parameters():
                parameter.requires_grad = False
        elif freeze != "none":
            n = int(freeze)
            # Freeze some params:
            for idx, parameter in enumerate(self.feature_extractor.named_parameters()):
                parameter[1].requires_grad = False
                if idx == n:
                    break

    def out_shape(self):
        if self.net == "vgg":
            # VGG:
            return (512, 14)
        elif self.net == "res":
            # Resnet:
            return (512, 14)
            
    def forward(self, x):
        features = self.feature_extractor(x)
        if self.net == "res":
            features = features.view(-1, 512, 14, 14)
        return features
    
class Decoder(nn.Module):
    def __init__(self, input_shp=(512, 14)):
        super().__init__()
        
        in_channels, size = input_shp 
        
        #TODO: try upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1 instead of Upsample
        mode = "conv" # "bilinear", "conv"
        
        def create_upsampling(mode, channels):
            if mode == "bilinear":
                return nn.Upsample(scale_factor=2, mode="bilinear")
            elif mode == "conv":
                return nn.ConvTranspose2d(channels, channels, 2, stride=2, dilation=1, padding=0, output_padding=0)
        
        block1 = nn.Sequential(nn.Conv2d(in_channels, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1),
                               nn.ReLU(),
                               create_upsampling(mode, 512)
                              )
        block2 = nn.Sequential(nn.Conv2d(512, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 512, 3, padding=1),
                               nn.ReLU(),
                               create_upsampling(mode, 512)
                              )
        block3 = nn.Sequential(nn.Conv2d(512, 256, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(256, 256, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(256, 256, 3, padding=1),
                               nn.ReLU(),
                               create_upsampling(mode, 256)
                              )
        block4 = nn.Sequential(nn.Conv2d(256, 128, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(128, 128, 3, padding=1), 
                               nn.ReLU(),
                               create_upsampling(mode, 128)
                              )
        block5 = nn.Sequential(nn.Conv2d(128, 512, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(512, 1, 3, padding=1), 
                               #nn.Sigmoid()
                              )
        self.generator = nn.Sequential(block1, block2, block3, block4, block5)
    
    def forward(self, x):
        x =  self.generator(x)
        # Apply Sigmoid if we are evaluating the model, else the BCEWithLogitsLoss takes care of it:
        #if not self.training:
        #    x = torch.sigmoid(x)
        return x


class EncoderDecoder(nn.Module):
    def __init__(self, net_type, freeze):
        super().__init__()
        self.encoder = Encoder(net_type, freeze, pretrained=True)
        embedding_shape = self.encoder.out_shape()
        self.decoder = Decoder(embedding_shape)
        
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
    
def stack_img_pred_and_fix(imgs, targets, preds, loss_type):
    use_means = means.view(1, 3, 1, 1)
    use_stds = stds.view(1, 3, 1, 1)
    imgs = imgs.mul_(use_stds).add_(use_means) #* 255
    if "BCE" in loss_type:
        preds = torch.sigmoid(preds)
    # turn to 3 channels:
    preds = torch.cat([preds] * 3, dim=1)
    targets = torch.cat([targets] * 3, dim=1)
    # calc new shape:
    shp = list(preds.shape)
    shp[0] *= 3
    # stack:
    stacked = torch.stack([imgs, targets, preds], dim=1)
    # reshape and return:
    return stacked.view(shp)

def pearson_loss(preds, targets):
    x = preds
    y = targets
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    cost = torch.mean(vx * vy) / (torch.sqrt(torch.mean(vx ** 2)) * torch.sqrt(torch.mean(vy ** 2)))
    #cost = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    #cost = vx * vy * torch.rsqrt(torch.sum(vx ** 2)) * torch.rsqrt(torch.sum(vy ** 2)))
    return (-1 * cost) + 1
            
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

net_type = "vgg"  # "res", "vgg"
freeze = "none" # "all", "N", "none"
lr = 0.00003
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoder(net_type, freeze)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=3, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
loss_type = "BCE+MSE"
bce_fcn = torch.nn.BCEWithLogitsLoss() 
kl_div_func = torch.nn.KLDivLoss()
kl_div = lambda preds, targets: kl_div_func(torch.log(preds / preds.sum()), targets / targets.sum())
mse_fcn = torch.nn.MSELoss()
bce_down_fcn = BCELossDownsampling()
if loss_type == "BCEDown":
    loss_fcn = bce_down_fcn
elif loss_type == "BCE":
    loss_fcn = bce_fcn
elif loss_type == "MSE":
    loss_fcn = mse_fcn
elif loss_type == "KL":
    loss_fcn = kl_div
elif loss_type == "pearson":
    loss_fcn = pearson_loss
elif loss_type == "KL+pearson":
    loss_fcn = lambda x, y: pearson_loss(x, y) + kl_div(x, y)
elif loss_type == "BCE+MSE":
    loss_fcn = lambda x, y: bce_down_fcn(x, y) + mse_fcn(torch.sigmoid(x), y)

os.makedirs("imgs", exist_ok=True)

use_apex = True
if use_apex:
    model, optimizer = amp.initialize(model, opt, opt_level="O1")

writer = SummaryWriter(comment=loss_type + net_type + "freeze" + freeze + str(train_batch_size) + "_" + str(lr))
epochs = 20
pbar = tqdm(total=epochs * len(train_dl))
best_loss = None
for epoch in range(epochs):
    
                
    # Eval:
    model.eval()
    inputs = []
    preds = []
    targets = []
    for val_sample in val_dl:
        img = val_sample["image"].to(device)
        fix = val_sample["fixation"].to(device)
        
        with torch.no_grad():
            pred = model(img)
        targets.append(fix.to("cpu").float())
        preds.append(pred.to("cpu").float())
        inputs.append(img.to("cpu").float())
    inputs = torch.cat(inputs)
    preds = torch.cat(preds)
    targets = torch.cat(targets)
    
    # Calc metrics:
    val_bce = bce_fcn(preds, targets).item()
    val_bce_down = bce_down_fcn(preds, targets).item()
    val_loss = loss_fcn(preds, targets).item()
    preds = torch.sigmoid(preds)
    val_kl = kl_div(preds, targets).item()
    val_mse = mse_fcn(preds, targets).item()
    val_pearson = pearson_loss(preds, targets).item()
    writer.add_scalar("val/BCE", val_bce, epoch)
    writer.add_scalar("val/BCE_downsampled", val_bce_down, epoch)
    writer.add_scalar("val/loss", val_loss, epoch)
    writer.add_scalar("val/KL_div", val_kl, epoch)
    writer.add_scalar("val/MSE", val_mse, epoch)
    writer.add_scalar("val/pearsonCC", val_pearson, epoch)
    
    # Save images:
    save_imgs = stack_img_pred_and_fix(inputs, targets, preds, loss_type)
    torchvision.utils.save_image(save_imgs[:81], "imgs/" + str(epoch) + "_val_preds.png", nrow=9)
    
    model.train()
    total_train_loss = 0
    for batch in train_dl:
        pbar.update(1)
        img, fixation = batch["image"].to(device), batch["fixation"].to(device)
        
        pred = model(img)
        if "BCE" not in loss_type:
            pred = torch.sigmoid(pred)
        loss = loss_fcn(pred, fixation)
        
        if use_apex:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        opt.step()
        opt.zero_grad()
        
        total_train_loss += loss.detach().item()
    
    #pbar.set_postfix({"loss": total_train_loss / len(train_dl), "val_loss": val_loss})
    print("loss: ", round(total_train_loss / len(train_dl), 4), " val_loss: ", round(val_loss, 4))
    save(epoch, model, opt, val_loss, "newest_model.pt")
    if best_loss is None or val_loss <= best_loss:
        best_loss = val_loss
        save(epoch, model, opt, loss, "best_model.pt")
    
    # Change learning rate:
    scheduler.step(val_loss)
        
    
   
pbar.close()

# Save images:
with torch.no_grad():
    imgs = torch.cat([model(sample["image"].to(device)) for sample in val_dl][:-1])
    torchvision.utils.save_image(imgs[:100], "imgs/final_val_preds.png", nrow=10)
      