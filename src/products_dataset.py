import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


class ProductsDataset(Dataset):

    def __init__(self, xlsx_filepath, root_dir, transform=None):
        self.data_table = self._parse_data_table(xlsx_filepath)
        self.root_dir = root_dir
        self.transform = transform

    def _parse_data_table(self, xlsx_filepath):
        """
        Filter empty values from data table rows
        """
        xlsx_file = pd.ExcelFile(xlsx_filepath)
        df = xlsx_file.parse()
        df = df.loc[:, ['id', 'condition', 'category']].dropna()
        df = df.reset_index(drop=True)
        return df

    def __len__(self):
        return len(self.data_table)

    def __getitem__(self, index):
        data_row = self.data_table.loc[index]
        image_id = data_row['id']
        image_name = '{}.jpg'.format(image_id)
        full_name = os.path.join(self.root_dir, image_name)

        sample = None
        with Image.open(full_name) as image:
            if self.transform:
                image = self.transform(image)

            sample = {
                'image': np.array(image, dtype=np.uint8),
                'category': data_row['category'],
                'condition': data_row['condition']
            }

        return sample
