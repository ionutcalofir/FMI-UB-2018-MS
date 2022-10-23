import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

class BrainHemorrhageDataset(Dataset):
  def __init__(self, config, root_dir, transform=None, submission=False):
    self.root_dir = root_dir
    self.transform=transform
    self.submission = submission

    self.imgs = []
    self.labels = []
    with open(config, 'r') as f:
      for line in f:
        if self.submission:
          self.imgs.append(line.strip())
        else:
          self.imgs.append(line.strip().split()[0])
          self.labels.append(line.strip().split()[1])

  def __len__(self):
    return len(self.imgs)

  def __getitem__(self, idx):
    img_name = os.path.join(self.root_dir, self.imgs[idx])
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if self.transform:
      img = self.transform(img)

    if self.submission:
      return img, img_name

    return img, int(self.labels[idx])

if __name__ == '__main__':
  config = './data/configs/train/train.txt'
  root_dir = './data/images'

  import torchvision.transforms as transforms
  transform = transforms.Compose(
        [
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )
  ds = BrainHemorrhageDataset(config, root_dir, transform=transform)
  result = ds[5]

  import pdb; pdb.set_trace()
