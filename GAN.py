import numpy as np
import math
import os
import cv2
from tqdm import tqdm

# import lib for img processing
import torchvision.transforms as transforms
from torchvision.utils import save_image

# import lib for data processing
from torch.utils.data import DataLoader
from torchvision import datasets

#　in this tutorial, i implement pytorch
import torch.nn as nn
import torch.nn.functional as F
import torch

import matplotlib.pyplot as plt
from natsort import natsorted 

n_epochs = 250   #epochs
batch_size = 64   #batch size  # I recommend that more than 32 
lr_rate = 0.0002   #learning rate
beta1 = 0.5     #learning rate beta1
beta2 = 0.999    #beta2
img_size = 64    #image size
channels = 3    #MNIST is gray scale dataset so is 1. If you want to challenge colorful dataset you should change it to 3
img_shape = (channels, img_size, img_size) # gray scale (1,28,28)
img_pixel = int(np.prod(img_shape)) # total pixel is 1*28*28
z_dim = 128     #z_dim is latent_dim
interval = 500    # How often to sample
video_interval = 2000

# First of all, we create an folder to pack dataset

#os.makedirs("./data/cats", exist_ok = True)

# Then download mnist through torch
DATA_DIR = './data'
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

train_ds = datasets.ImageFolder(DATA_DIR, transform=transforms.Compose([transforms.Resize(img_size),
                                                                        transforms.CenterCrop(img_size),
                                                                        transforms.ToTensor(),
                                                                        transforms.Normalize(*stats)
                                                                        ]
                                                                       )
                                )

dataloader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)

print(len(train_ds))
print(len(dataloader))
"""
dataloader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "./data/cat",
        train=True,
        download=True,
        transform = transforms.Compose(
            [transforms.Resize(img_size),      # resize to 28*28
              transforms.ToTensor(),         # to tensor
              transforms.Normalize([0.5], [0.5])]   # normalize img  #([0.5], [0.5], [0.5])
        ),
    ),
    batch_size = batch_size,
    shuffle = True,
)
"""

# creat a Discriminator
class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()

    self.FC = nn.Sequential(
        nn.Linear(img_pixel, 512),  # img_pixel is 784  
        nn.LeakyReLU(0.2, inplace=True), # limit data to 0~1 to avoid Vanishing gradient problem
        nn.Linear(512, 256),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(256, 128),
        nn.LeakyReLU(0.2, inplace=True),
        nn.Linear(128, 1),
        nn.Sigmoid(),         # before classify, use sigmoid.
    )

  def forward(self, img):
    flat = img.view(img.size(0), -1)
    validity = self.FC(flat)

    return validity

# great! we have a disicriminator to catch the fake data


# create a generator
class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()

    # There are lots of same type block, so i prefer define a function to do the same thing.
    # Then, just simply up scale to lots of pixel.
    # Finally turn it into the correct image size.
    self.FCin = nn.Linear(z_dim, 128)
    self.FC1 = self._block(128, 256)
    self.FC2 = self._block(256, 512)
    self.FC3 = self._block(512, 1024)
    self.FCout = nn.Sequential(
        nn.Linear(1024, img_pixel),
        nn.Tanh(),
    )
      

  # hidden layer
  def _block(self, in_fea, out_fea):
    return nn.Sequential(
        nn.Linear(in_fea, out_fea),
        nn.BatchNorm1d(out_fea, 0.8),  # google it
        nn.LeakyReLU(0.2, inplace=True),
    )

  # put random z into layer sequentially.
  def forward(self, z):
    flat = z.view(z.size(0), -1)
    fea = self.FCin(flat)
    fea = self.FC1(fea)
    fea = self.FC2(fea)
    fea = self.FC3(fea)
    fake_img = self.FCout(fea)
    fake_img = fake_img.view(fake_img.size(0), *img_shape)
    return fake_img

# great! we have a cool generator now.


# The loss we only need is Binary Cross Entropy Loss! I bet you didn't expect that!

adversarial_loss = torch.nn.BCELoss()


# at first we need to get our gpu!
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device('cpu')

# call the generator and discriminator
generator = Generator()
discriminator = Discriminator()

# put them into your device
if torch.cuda.is_available():
    generator = generator.to(device)
    discriminator = discriminator.to(device)

# Create Optimizer
# In this tutorial we use Adam
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=(beta1, beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr_rate, betas=(beta1, beta2))

# create a folder to save image
os.makedirs("./results", exist_ok = True)
result_dir = './results'
os.makedirs("./images", exist_ok = True)
sample_dir = './images'


"""
for batch_id, (img, _) in enumerate(tqdm(dataloader)):
    save_image(img, "images/%d.png" %batch_id,nrow=8, normalize=True)
"""
def result(g_losses, d_losses, real_scores, fake_scores):
    with torch.no_grad():
        loss_data = plt.figure()
        plt.plot(g_losses)
        plt.plot(d_losses,'r')
        plt.xlabel('epoch')
        plt.legend(['G','D'])
        loss_data.savefig(os.path.join(result_dir, 'loss_figure'))
        
        score_data = plt.figure()
        plt.plot(real_scores)
        plt.plot(fake_scores,'r')
        plt.xlabel('epoch')
        plt.legend(['R','F'])
        score_data.savefig(os.path.join(result_dir, 'score_figure'))
        
    files = [os.path.join(sample_dir, f) for f in os.listdir(sample_dir)]
    files = natsorted(files)

    output_video ='trainning.mp4'
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 1
    output_path = os.path.join(result_dir, output_video)
    out = cv2.VideoWriter(output_path, fourcc, fps, (530,530))
    for fname in files:
        str_index_1 = fname.index("\\")
        str_index_2 = fname.index(".", 1)
        str_number = fname[str_index_1+1:str_index_2]
        if int(str_number) % video_interval == 0:
            out.write(cv2.imread(fname))
    out.release()  


# ------------------
# Training Start
# ------------------
g_losses = []
d_losses = []
real_scores = []
fake_scores = []

g_losses.append(0)
d_losses.append(0)
real_scores.append(1)
fake_scores.append(0)


for epoch in range(n_epochs):
    total_g_loss = 0
    total_d_loss = 0
    real_batch_scores = []
    fake_batch_scores = []
    for batch_id, (img, _) in enumerate(tqdm(dataloader)):
  
        # throw our img into gpu
        real_imgs = img.to(device)
        
        # create label to caculate BCE
        _valid = torch.ones(size=(real_imgs.shape[0], 1), requires_grad=False).to(device) # (batch_size, 1) [1, 1, 1,...]
        _fake = torch.zeros(size=(real_imgs.shape[0], 1), requires_grad=False).to(device) # (batch_size, 0) [0, 0, 0,...]
        
        
        # -----------------
        # Train Generator
        # -----------------
        optimizer_G.zero_grad()
        
        # Sample noise
        z = torch.randn(real_imgs.shape[0], z_dim, 1, 1).to(device)  # (batch_size, z_dim, 1, 1)
        
        # generate images!
        gen_imgs = generator(z)
        
        # Generator :min log(1 - D(G(z))) <-> max log(D(G(z))
        preds = discriminator(gen_imgs)
        g_loss = adversarial_loss( preds, _valid)
        
        g_loss.backward()
        optimizer_G.step()
        
        # --------------------
        # Train Discriminator
        # --------------------
        optimizer_D.zero_grad()
        
        # Discriminator: max log(D(x)) + log(1 - D(G(z)))
        real_preds = discriminator(real_imgs)
        real_loss = adversarial_loss(real_preds, _valid)
        real_score = torch.mean(real_preds).item()
        
        fake_preds = discriminator(gen_imgs.detach())
        fake_loss = adversarial_loss(fake_preds, _fake)
        fake_score = torch.mean(fake_preds).item()
        
        d_loss = (real_loss + fake_loss) / 2
        
        d_loss.backward()
        optimizer_D.step()
        
        total_g_loss += g_loss.cpu()
        total_d_loss += d_loss.cpu()
        #g_losses.append(g_loss.cpu())
        #d_losses.append(d_loss.cpu())
        real_batch_scores.append(real_score)
        fake_batch_scores.append(fake_score)
        
        print(
          "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
          % (epoch, n_epochs, batch_id, len(dataloader), d_loss.item(), g_loss.item())
        )
        
        batches_done = epoch * len(dataloader) + batch_id
        if batches_done % interval == 0:
          save_image(gen_imgs.data[:64], "images/%d.png" % batches_done, nrow=8, normalize=True)
    g_losses.append(total_g_loss/len(dataloader))
    d_losses.append(total_d_loss/len(dataloader))
    real_scores.append(np.mean(real_batch_scores))
    fake_scores.append(np.mean(fake_batch_scores))
          
result(g_losses, d_losses, real_scores, fake_scores)

 