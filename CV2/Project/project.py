import os
import random

from tqdm import tqdm
import numpy as np
import imageio
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, Dataset
import torchvision
from torchvision import transforms
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
        
        mode = "bilinear" # "bilinear", "conv"
        
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
        block5 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), 
                               nn.ReLU(),
                               nn.Conv2d(64, 1, 3, padding=1), 
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
    def __init__(self, root_dir, image_file, fixation_file, input_transform=None, target_transform=None, train_transform=None, train_input_transform=None):
        self.root_dir = root_dir
        self.image_files = read_file(os.path.join(root_dir, image_file))
        if fixation_file is not None:
            self.fixation_files = read_file(os.path.join(root_dir, fixation_file))
            assert(len(self.image_files) == len(self.fixation_files))
        else:
            self.fixation_files = None
        self.input_transform = input_transform
        self.target_transform = target_transform
        self.train_transform = train_transform
        self.train_input_transform = train_input_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        # Read input img:
        img_name = os.path.join(self.root_dir, self.image_files[idx])
        image = imageio.imread(img_name)

        # Read target img:
        sample = {'image': image}
        if self.fixation_files is not None:
            fix_name = os.path.join(self.root_dir, self.fixation_files[idx])
            fix = imageio.imread(fix_name)
            sample['fixation'] = fix
            
        # Apply transforms:
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        if self.train_transform:
            # Set random seed to same val to have same transform for input and target:
            random.seed(seed)
            sample["image"] = self.train_transform(sample["image"])
            random.seed(seed)
            sample["fixation"] = self.train_transform(sample["fixation"])
        if self.train_input_transform:
            sample["image"] = self.train_input_transform(sample["image"])
        if self.input_transform:
            sample["image"] = self.input_transform(sample["image"])
        if self.fixation_files is not None and self.target_transform:
            sample["fixation"] = self.target_transform(sample["fixation"])
            
        sample["img_name"] = img_name
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
    imgs = imgs.mul_(use_stds).add_(use_means)
    #if "BCE" in loss_type:
    #    preds = torch.sigmoid(preds)
    # turn to 3 channels:
    preds = torch.cat([preds] * 3, dim=1)
    if targets is not None:
        targets = torch.cat([targets] * 3, dim=1)
        stack_list = [imgs, targets, preds]
    else:
        stack_list = [imgs, preds]
        
    # calc new shape:
    shp = list(preds.shape)
    shp[0] *= len(stack_list)
    # stack:
    stacked = torch.stack(stack_list, dim=1)
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
            
# Create transforms:
means = torch.tensor([0.485 , 0.456 , 0.406])
stds = torch.tensor([0.229 , 0.224 , 0.225])
normalize = transforms.Normalize(mean=means, std=stds)
input_transform = transforms.Compose([transforms.ToTensor(), normalize])
target_transform = transforms.Compose([transforms.ToTensor()])
train_transform = transforms.Compose([transforms.ToPILImage(), transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(5, resample=False, expand=False, center=None, fill=None)])
train_input_transform = transforms.Compose([transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)])#, torchvision.transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0, inplace=False)])
# Create datasets:
ds = FixationDataset("data", "train_images.txt", "train_fixations.txt", input_transform, target_transform, train_transform, train_input_transform)      
val_ds = FixationDataset("data", "val_images.txt", "val_fixations.txt", input_transform, target_transform)  
test_ds = FixationDataset("data", "test_images.txt", None, input_transform, None)  

sampler = RandomSampler(ds)
torch.backends.cudnn.benchmark = True
pin_mem = False
train_batch_size = 16
val_batch_size = 9
train_dl = DataLoader(ds, batch_size=train_batch_size, num_workers=2, pin_memory=pin_mem, sampler=sampler)
val_dl = DataLoader(val_ds, batch_size=val_batch_size, num_workers=2, pin_memory=pin_mem)
test_dl = DataLoader(test_ds, batch_size=val_batch_size, num_workers=2, pin_memory=pin_mem)

net_type = "res"  # "res", "vgg"
freeze = "none" # "all", "N", "none"
lr = 0.0003
epochs = 50
loss_type = "BCEDown"
mse_alpha = 1.1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EncoderDecoder(net_type, freeze)
model.to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=1, verbose=True, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
bce_fcn = torch.nn.BCEWithLogitsLoss() 
kl_div_func = torch.nn.KLDivLoss()
kl_div = lambda preds, targets: kl_div_func(torch.log(preds / preds.sum()), targets / targets.sum())
mse_fcn = torch.nn.MSELoss()
mse_unreduced = torch.nn.MSELoss(reduce="none")
def weighted_mse(x, y):
    losses = mse_unreduced(x, y)
    weights = (mse_alpha - y) ** 2
    return (losses / weights).mean() 
bce_down_fcn = BCELossDownsampling()
if loss_type == "BCEDown":
    loss_fcn = bce_down_fcn
elif loss_type == "BCE":
    loss_fcn = bce_fcn
elif loss_type == "MSE":
    loss_fcn = lambda x, y: mse_fcn(torch.sigmoid(x), y)
elif loss_type == "KL":
    loss_fcn = lambda x, y: kl_div(torch.sigmoid(x), y)
elif loss_type == "pearson":
    loss_fcn = lambda x, y: pearson_loss(torch.sigmoid(x), y)
elif loss_type == "KL+pearson":
    loss_fcn = lambda x, y: pearson_loss(torch.sigmoid(x), y) + kl_div(torch.sigmoid(x), y)
elif loss_type == "BCE+MSE":
    loss_fcn = lambda x, y: bce_down_fcn(x, y) + mse_fcn(torch.sigmoid(x), y)
elif loss_type == "WeightedMSE":
    loss_fcn = lambda x, y: weighted_mse(torch.sigmoid(x), y)
    
test = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.8, 1.0]])
#print(loss_type, "pred for same x and y of: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]", loss_fcn(test, test))
#print(loss_type, "pred for x and 1 - x of: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]", loss_fcn(test, 1 - test))
#print(loss_type, "pred for x=0 and y is: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]", loss_fcn(test, torch.zeros_like(test)))
#print(loss_type, "pred for x=1 and y: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]", loss_fcn(test, torch.ones_like(test)))

os.makedirs("imgs", exist_ok=True)

use_apex = True
if use_apex:
    model, optimizer = amp.initialize(model, opt, opt_level="O1")

writer = SummaryWriter(comment=loss_type + net_type + "freeze" + freeze + str(train_batch_size) + "_" + str(lr))
pbar = tqdm(total=epochs * len(train_dl))
best_loss = None
loss_dict = {"val": [], "train": []}
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
    #print(preds.mean().item(), preds.std().item(), preds.min().item(), preds.max().item())
    if "BCE" in loss_type:
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
    
        pbar.set_postfix({"loss": loss.detach().item()})
    print("loss: ", round(total_train_loss / len(train_dl), 4), " val_loss: ", round(val_loss, 4))
    save(epoch, model, opt, val_loss, "newest_model.pt")
    if best_loss is None or val_loss <= best_loss:
        best_loss = val_loss
        save(epoch, model, opt, loss, "best_model.pt")
        
    loss_dict["train"].append(total_train_loss / len(train_dl))
    loss_dict["val"].append(val_loss)
    
    # Change learning rate:
    scheduler.step(val_loss)
        
    
   
pbar.close()

# Save images:
with torch.no_grad():
    imgs = torch.cat([model(sample["image"].to(device)) for sample in val_dl][:-1])
    torchvision.utils.save_image(imgs[:100], "imgs/final_val_preds.png", nrow=10)
      
# Save test image predictions:
test_folder = "test_preds/"
os.makedirs(test_folder, exist_ok=True)
with torch.no_grad():
    for sample in test_dl:
        img = sample["image"].to(device)
        img_names = sample["img_name"]
        preds = model(img).to("cpu").float()
        if "BCE" in loss_type:
            preds = torch.sigmoid(preds)
        for idx, pred in enumerate(preds):
            img_name = img_names[idx]
            img_name = img_name[img_name.find("image-"):]
            pred_name = "prediction" + img_name[5:]
            torchvision.utils.save_image(pred, test_folder + pred_name)
            
# Save some test img predictions next to targets of best model:
model.load_state_dict(torch.load("best_model.pt")["model"])
model = model.to(device)
model.eval()
os.makedirs("imgs_test", exist_ok=True)
inputs = []
preds = []
for val_sample in test_dl:
    img = val_sample["image"].to(device)
        
    with torch.no_grad():
        pred = model(img)
    if "BCE" in loss_type:
        pred = torch.sigmoid(pred)
    preds.append(pred.to("cpu").float())
    inputs.append(img.to("cpu").float())
inputs = torch.cat(inputs)
preds = torch.cat(preds)
save_imgs = stack_img_pred_and_fix(inputs, None, preds, loss_type)
torchvision.utils.save_image(save_imgs[:80], "imgs_test/" + "test_preds.png", nrow=8)


# Plot losses:
import matplotlib.pyplot as plt
import seaborn as sns
train_losses = loss_dict["train"]
val_losses = loss_dict["val"]
steps = range(len(train_losses))
sns.lineplot(steps, train_losses, label="Train")
sns.lineplot(steps, val_losses, label="Validation")
plt.legend()
plt.title("Losses during Training")
plt.xlabel("Epochs")
plt.ylabel("BCE Loss")
plt.savefig("losses.pdf")