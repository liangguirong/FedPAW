# encoding: utf-8
"""
Read images and corresponding labels.
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
N_CLASSES = 7
CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis']
class CheXpertDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None,strong_transform=None):
        """
        Args:
            data_dir: path to image directory.
            csv_file: path to the file containing images
                with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(CheXpertDataset, self).__init__()
        file = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.images = file['ImageID'].values
        self.labels = file.iloc[:, 1:].values.astype(int)
        self.labels_id = self.labels
        _, self.labels = torch.max(torch.tensor(self.labels), dim=-1)
        self.transform = transform
        self.strong_transform = strong_transform
        print('Total # images:{}, labels:{}'.format(len(self.images),len(self.labels)))

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        items = self.images[index]
        if '.png' not in self.images[index]:
            image_name = os.path.join(self.root_dir, self.images[index] + '.jpg')
        else:
            image_name = os.path.join(self.root_dir, self.images[index])
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        labels_id = self.labels_id[index]
        if self.transform is not None:
            image1 = self.transform(image)
            image3 = self.transform(image)
        if self.strong_transform is not None:
            image2 = self.strong_transform(image)
        else:
            image2 = self.transform(image)
        return items, index, image1,image2,image3,label,labels_id
    def __len__(self):
        return len(self.images)
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        items, index, image, image2,image3,label,labels_id= self.dataset[self.idxs[item]]
        return image,label,image2,image3