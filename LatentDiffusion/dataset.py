import pandas as pd
import torchvision.transforms as T

from PIL import Image
from torch.utils.data import Dataset

class Uncond_Dataset(Dataset):
    def __init__(self, path, image_size):
        data = pd.read_csv(path)

        self.image_paths = data["Image Path"]
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('RGB').resize((self.image_size, self.image_size))
        img = T.ToTensor()(img) * 2.0 - 1.0

        return img

    def __getitem__(self, idx):
        image = self.preprocess_image(self.image_paths[idx])

        return image