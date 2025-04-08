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
        
        # Define the layers for the generator
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=7, stride=1, padding=3),
            nn.Tanh()  # normalized image
        )
    
    def forward(self, x):
        return self.model(x)

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



