import os

import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from corrupted_files import CORRUPTED_FILES

ImageFile.LOAD_TRUNCATED_IMAGES = True

class ImageDataset(Dataset):

    def __init__(self,
                 df,
                 images_dir,
                 n_classes,
                 label_dict,
                 has_answer=True,
                 transforms=None):
        """
        :param df: Pandas dataframe
        :param images_dir: Images directory
        :param n_classes: Total number of classes (in the entire dataset - train and test)
        :param label_dict: Mapping of category id with the corresponding category id in range(0, n_classes)
        :param has_answer: Whether the dataset contains answers
        :param transforms: Image transformations to be applied to each image
        """
        self.df = df
        self.images_dir = images_dir
        self.add_columns()
        self.n_classes = n_classes
        self.label_dict = label_dict
        self.has_answer = has_answer
        self.transforms = transforms
        self.deleteRows()

    def add_columns(self):
        self.df['image_path'] = self.df['file_name'].apply(
            lambda x: self.images_dir + '/' + x)
    
    def deleteRows(self):
        self.df = self.df[~self.df['file_name'].isin(CORRUPTED_FILES)]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        cur_idx_row = self.df.iloc[idx]
        img_id = cur_idx_row['file_name']
        img_path = cur_idx_row['image_path']

        img = Image.open(img_path)

        if self.transforms is not None:
            img = self.transforms(img)

        if not self.has_answer:
            return img, img_id

        label = torch.zeros((self.n_classes,), dtype=torch.float32)
        category_id = self.label_dict[cur_idx_row['category_id']]
        label[category_id] = 1.0

        return img, label

    def get_df(self):
        return self.df
