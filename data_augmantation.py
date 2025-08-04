import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from customDataset import CatsAndDogsDataset

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

dataset = CatsAndDogsDataset(root_dir="data", transform=transform)

image, label = dataset[0]
save_image(image, "sample_image.png")
