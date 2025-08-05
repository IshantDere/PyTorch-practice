import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import CatsAndDogsDataset

my_transforms = transforms.Compose([
    transforms.ToPILImage()
    transforms.Resize((256,256))
    transforms.RandomCrop((224,334))
    transforms.ColorJitter(brightness = 0.5)
    transforms
])