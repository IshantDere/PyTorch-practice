import torch
from torch import nn
from torch import optim
import PIL as Image
from torchvision import transforms
from torchvision import models
from torchvision.utils import save_image

model = models.vgg19(pretrained = True).features
print(model)

# from the layers we want :- [0, 5, 10, 19, 28]

class VGG(nn.Module):
    def __init__(self):
        super().__init__()
        self.choosen_layer = ['0', '5', '10', '19', '28']
        self.model = models.vgg19(pretrained = True).features[:29]

    def forward(self, x):
        features = []

        for layer_num, layer in enumerate(self.model):
            features.append(x)

        return features

def load_image(image_name):
    image = Image.open(image_name)
    image = loader(image).unsqeeze(0)

    return image.to(device)

device = torch.device("cuda" if torch.cuda.is_available else "cpu")
image_size = 356

loader = transforms.Compose(
    [
        transforms.resize((image_size, image_size))
        transforms.ToTensor()
    ]
)

original_image = load_image()
style_image = load_image()

generated = original_image.clone().requires_grads_(True)

# hyperparameters

total_steps = 6000
learning_rate = 0.001
alpha = 1
beta = 0.01
optimizer = optim.Adam([generated], lr = learning_rate)