import torch
import torch.nn as nn
import torch.optim as optim
import itertools
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        # Downsampling (encoder)
        self.down1 = self.conv_block(3, 64, stride=2)
        self.down2 = self.conv_block(64, 128, stride=2)
        self.down3 = self.conv_block(128, 256, stride=2)
        self.down4 = self.conv_block(256, 512, stride=2)

        # Upsampling (decoder)
        self.up1 = self.upconv_block(512, 256)
        self.up2 = self.upconv_block(256, 128)
        self.up3 = self.upconv_block(128, 64)
        self.up4 = self.upconv_block(64, 3, final_layer=True)
    
    def conv_block(self, in_channels, out_channels, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(out_channels)
        )
    

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()  # binary output (real/fake)
        )
    
    def forward(self, x):
        return self.model(x)


# Loss Functions

# 1. Adversarial loss (real/fake)

# Binary Cross-Entropy Loss (for Discriminators)
criterion = nn.BCELoss()

def adversarial_loss(D, real, fake):
    real_loss = criterion(D(real), torch.ones_like(D(real)))
    fake_loss = criterion(D(fake), torch.zeros_like(D(fake)))
    return (real_loss + fake_loss) / 2

# 2. Cycle Consistency Loss
def cycle_loss(real_image, reconstructed_image):
    return nn.L1Loss()(real_image, reconstructed_image)

# 3. Identity Loss
def identity_loss(real_image, generated_image):
    return nn.L1Loss()(real_image, generated_image)



