import torch
import torchvision.models as models, transforms
from PIL import Image

vgg = models.vgg19(pretrained=True).features

content = torch.randn(1, 3, 224, 224)
style   = torch.randn(1, 3, 224, 224)

target = content.clone().requires_grad_(True)

optimizer = torch.optim.Adam([target], lr=0.01)

for step in range(100):
    target_features  = vgg(target)
    content_features = vgg(content)
    style_features   = vgg(style)

    content_loss = ((target_features - content_features)**2).mean()
    style_loss   = ((target_features - style_features)**2).mean()
    loss = content_loss + style_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()