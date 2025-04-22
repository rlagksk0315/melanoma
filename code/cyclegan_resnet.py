import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from PIL import Image

class ResnetBlock(nn.Module):
    def __init__(self, dim):
        super(ResnetBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )
    
    def forward(self, x):
        return x + self.block(x)  # skip connection


class ResnetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, n_blocks=9):
        super(ResnetGenerator, self).__init__()
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, 64, kernel_size=7, padding=0),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features *= 2

        # Residual Blocks
        for _ in range(n_blocks):
            model += [ResnetBlock(in_features)]

        # Upsampling
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = out_features // 2

        # Output Layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    

# PatchGAN Discriminator
# The PatchGAN discriminator classifies whether each N x N patch in an image is real or fake
# It uses a series of convolutional layers followed by LeakyReLU activations and Instance Normalization
# The final layer outputs a single channel for each patch, indicating the probability of it being real
# or fake. The model is designed to work with images of size 256x256, but can be adapted for other sizes.
class PatchGANDiscriminator(nn.Module):
    def __init__(self, input_nc=3):
        super(PatchGANDiscriminator, self).__init__()
        model = [
            nn.Conv2d(input_nc, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]

        in_channels = 64
        out_channels_list = [128, 256, 512]
        for out_channels in out_channels_list:
            stride = 1 if out_channels == 512 else 2
            model += [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            in_channels = out_channels

        model += [
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1)  # No Sigmoid for LSGAN
        ]

        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)
    

# Image Buffer
# It stores a fixed number of images and randomly samples from them during training
# This helps to stabilize training by providing a diverse set of images for the discriminator
class ImageBuffer():
    def __init__(self, max_size=50):
        self.max_size = max_size
        self.buffer = []
    
    def push_and_pop(self, images):
        result = []
        for image in images:
            image = image.unsqueeze(0)
            if len(self.buffer) < self.max_size:
                self.buffer.append(image)
                result.append(image)
            else:
                if random.uniform(0,1) > 0.5:
                    idx = random.randint(0, self.max_size - 1)
                    result.append(self.buffer[idx].clone())
                    self.buffer[idx] = image
                else:
                    result.append(image)

        return torch.cat(result)

# Loss Functions

# 1. Adversarial loss (real/fake)
# LSGAN
mse = nn.MSELoss()
def lsgan_loss(real, fake):
    return mse(real, fake)

def generator_adversarial_loss(D, fake):
    pred_fake = D(fake)
    target_real = torch.ones_like(pred_fake)
    return lsgan_loss(pred_fake, target_real)

def discriminator_adversarial_loss(D, real, fake):
    pred_real = D(real)
    pred_fake = D((fake.detach()))
    target_real = torch.ones_like(pred_real)
    target_fake = torch.zeros_like(pred_fake)
    real_loss = lsgan_loss(pred_real, target_real)
    fake_loss = lsgan_loss(pred_fake, target_fake)
    
    return (real_loss + fake_loss) / 2
    

# 2. Cycle consistency loss
def cycle_loss(real, cycled):
    return nn.L1Loss()(real, cycled)

# 3. Identity loss
def identity_loss(real, identity):
    return nn.L1Loss()(real, identity)
