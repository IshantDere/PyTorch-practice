import torch
from torchvision import transforms
from PIL import Image

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  
    transforms.RandomRotation(20),            
    transforms.ColorJitter(brightness=0.2, contrast=0.2), contrast
    transforms.ToTensor()                     
])