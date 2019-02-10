import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ProductsDataset(Dataset):

    def __init__(self, xlsx_filepath, root_dir, transform=None):
        self._category_to_id = None
        self._condition_to_id = None
        self._data_table = None
        self._parse_data_table(xlsx_filepath)
        self._id_to_category = {cat_id: category for category, cat_id
                                in self._category_to_id.items()}
        self._id_to_condition = {cond_id: condition for condition, cond_id
                                 in self._condition_to_id.items()}

        self.root_dir = root_dir
        self.transform = transform

    def _parse_data_table(self, xlsx_filepath):
        """
        Filter empty values from data table rows
        """
        xlsx_file = pd.ExcelFile(xlsx_filepath)
        df = xlsx_file.parse()
        df = df.loc[:, ['id', 'condition', 'category']].dropna()
        self._data_table = df.reset_index(drop=True)

        categories = sorted(list(set(self._data_table.loc[:, 'category'])))
        self._category_to_id = {categories[i]: i for i in range(len(categories))}

        conditions = sorted(list(set(self._data_table.loc[:, 'condition'])))
        self._condition_to_id = {conditions[i]: i for i in range(len(conditions))}

    def get_category_by_id(self, cat_id):
        return self._id_to_category[cat_id]

    def get_condition_by_id(self, cond_id):
        return self._id_to_condition[cond_id]

    def __len__(self):
        return len(self._data_table)

    def __getitem__(self, index):
        data_row = self._data_table.loc[index]
        image_id = data_row['id']
        image_name = '{}.jpg'.format(image_id)
        full_name = os.path.join(self.root_dir, image_name)

        with Image.open(full_name) as image:
            sample = {
                'image': image,
                'category': self._category_to_id[data_row['category']],
                'condition': self._condition_to_id[data_row['condition']]
            }
            if self.transform:
                sample = self.transform(sample)

        return sample
