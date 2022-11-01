import os
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image

def getListOfFiles(dirName):
    
    listOfFile = os.listdir(dirName)
    allFiles = list()

    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

class DomainDataset(Dataset):
    """
    A data loader for multi-domain datasets.

    Args:
        dataset_dir (string): Root directory path.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.

     Attributes:
        domain_names (list): List of the domain names sorted alphabetically.
        class_names (list): List of the class names sorted alphabetically.
        ddict (dict): Dict with items (domain_name, domain_index).
        cdict (dict): Dict with items (class_name, class_index).
    """
    def __init__(
        self,
        dataset_dir,
        transform=None,
    ):
        self.transform = transform
        self.img_paths = getListOfFiles(dataset_dir)
        
        self.domain_names = np.sort(os.listdir(dataset_dir))
        self.class_names = np.sort(os.listdir(os.path.join(dataset_dir, self.domain_names[0])))

        self.ddict = {self.domain_names[i]:i for i in range(len(self.domain_names))}
        self.cdict = {self.class_names[i]:i for i in range(len(self.class_names))}        

    def _rgb_loader(self, path):
        with open(path, "rb") as f:
            with Image.open(f) as img:
                return img.convert("RGB")

    def __getitem__(self, index):

        path = self.img_paths[index]
        target = self.cdict[path.split('/')[-2]]
        domain = self.ddict[path.split('/')[-3]]

        img = self._rgb_loader(path)

        if self.transform is not None:
            img = self.transform(img)
      
        return img, target, domain
      
    def __len__(self):
        return len(self.img_paths)
