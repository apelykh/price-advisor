import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ProductsDataset(Dataset):

    def __init__(self, xlsx_filepath, root_dir, transform=None):
        self._classes_to_id = {}
        self._id_to_classes = {}
        self._data_table = None
        self._parse_data_table(xlsx_filepath)
        self.classes_weights = self._get_classes_weights()

        self.root_dir = root_dir
        self.transform = transform

    def _parse_data_table(self, xlsx_filepath, fields_to_keep=('id', 'condition', 'category')):
        """
        Filter empty values from data table rows
        """
        xlsx_file = pd.ExcelFile(xlsx_filepath)
        df = xlsx_file.parse()
        df = df.loc[:, fields_to_keep].dropna()
        self._data_table = df.reset_index(drop=True)

        for field in fields_to_keep:
            # TODO: get rid of the dirty hack
            if field == 'id':
                continue
            class_list = sorted(list(set(self._data_table[field])))
            self._classes_to_id[field] = {class_list[i]: i for i in range(len(class_list))}
            self._id_to_classes[field] = {value: key for key, value in self._classes_to_id[field].items()}

    def _get_classes_weights(self, fields=('category', 'condition')):
        """
        Compute the weights for weighted loss application.
        :param fields: sequence of features; the weights will be computed for all the classes of each feature.
        :return:
        """
        df = self._data_table
        class_weights = {}

        for field in fields:
            class_weights[field] = {}
            class_list = sorted(list(set(df[field])))
            num_images_by_class = [sum(df[field] == c) for c in class_list]
            max_num_images = max(num_images_by_class)

            for i, c in enumerate(class_list):
                cur_num_images = num_images_by_class[i]
                class_id = self._classes_to_id[field][c]
                class_weights[field][class_id] = max_num_images / cur_num_images

        return class_weights

    def get_class_by_id(self, class_id, field):
        return self._id_to_classes[field][class_id]

    def __len__(self):
        return len(self._data_table)

    def __getitem__(self, index):
        data_row = self._data_table.loc[index]
        image_id = data_row['id']
        image_name = '{}.jpg'.format(image_id)
        full_name = os.path.join(self.root_dir, image_name)

        with Image.open(full_name) as image:
            if self.transform:
                image = self.transform(image)

        sample = {
            'image': image,
            'category': self._classes_to_id['category'][data_row['category']],
            'condition': self._classes_to_id['condition'][data_row['condition']]
        }

        return sample


# dataset = ProductsDataset(xlsx_filepath='../data/products.xlsx',
#                           root_dir='../data/images')