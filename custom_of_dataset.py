import torch
import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from skimage import io

class CatsVsDogs(Dataset):
    def __init__(self, csv_file, root_dir,transform = None):
        self.annotation = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotation)

    def ___getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotation.iloc[index, 0])
        image = io.imread(img_path)
        y_lable = torch.tensor(int(self.annotation.ilco[index, 0]))

        if self.transform:
            image = self.transform(image)

        return (image, y_lable)