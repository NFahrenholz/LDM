import pandas as pd
import torchvision.transforms as T
import xml.etree.ElementTree as ET

import cv2
import torch

from PIL import Image
from torch.utils.data import Dataset

from config import config

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

def get_bboxes(root):
    bboxes = []
    for bbox in root.iter('bbox'):
        label = int(bbox.find('label_id').text)

        x1 = int(bbox.find('x1').text)
        y1 = int(bbox.find('y1').text)
        x2 = int(bbox.find('x2').text)
        y2 = int(bbox.find('y2').text)

        bboxes.append((x1, y1, x2, y2, label))

    return bboxes

class Layout_Dataset(Dataset):
    def __init__(self, path, image_size):
        data = pd.read_csv(path)

        self.image_paths = data["Image Path"]
        self.image_names = data["Image Name"]
        self.image_size = image_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        tree = ET.parse(f"../data/BoundingBoxes/{self.image_names[idx]}.xml")
        root = tree.getroot()

        bboxes = get_bboxes(root)

        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        transformed = config.transform(image=img, bboxes=bboxes)
        img_transformed = T.ToTensor()(transformed['image']) * 2.0 - 1.0

        bboxes = torch.zeros((20, 5), dtype=torch.int)
        bboxes[:] = torch.IntTensor([0, 0, 0, 0, 36])
        bboxes = bboxes.flatten()
        bboxes[:(len(transformed['bboxes']*5))] = torch.IntTensor(transformed['bboxes']).flatten()

        return img_transformed, bboxes