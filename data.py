import torch
from torch import nn
from torch.utils import data
import torchvision
from config import Config
class GeoDataset(data.Dataset):
    def __init__(self,df,eval=False):
        self.df = df.reset_index(drop=True)
        self.aug = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(Config.IMG_SIZE),
                                                torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
                                                torchvision.transforms.RandomHorizontalFlip(),
                                                torchvision.transforms.RandomRotation(20),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ])
        if eval:
            self.aug = torchvision.transforms.Compose([
                                                torchvision.transforms.Resize(Config.IMG_SIZE),
                                                torchvision.transforms.ToTensor(),
                                                torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                            ]) 

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self,index):
        # print(index)
        img_path = self.df.loc[index,"files"]
        img = Image.open(img_path).convert("RGB").crop((0,0,IMG_SIZE,IMG_SIZE))
        label = self.df.loc[index,"target"]

        img = self.aug(img)
        return img,torch.tensor(label,dtype=torch.long)