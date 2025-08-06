import cv2
import albumentations as A
import numpy as np
from utils import plot_examples
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.transform = A.Compose([
            A.Resize(128, 128),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        return image, 0  

dataset = CustomDataset("path/to/images")
loader = DataLoader(dataset, batch_size=4, shuffle=True)

for imgs, labels in loader:
    print(imgs.shape)  
    break