import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class DogBreedDataset(Dataset):
    def __init__(self, img_dir, labels_csv, transform=None):
        self.img_dir = img_dir
        self.df = pd.read_csv(labels_csv)
        self.transform = transform
        self.breeds = sorted(self.df['breed'].unique())
        self.label_map = {b: i for i, b in enumerate(self.breeds)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['id'] + '.jpg')
        image = Image.open(img_path).convert('RGB')
        label = self.label_map[row['breed']]
        if self.transform:
            image = self.transform(image)
        return image, label