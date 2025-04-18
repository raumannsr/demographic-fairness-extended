import os
import os.path

import pandas as pd
import torch.utils.data as data
from torchvision.datasets.folder import default_loader


class CSVDataset(data.Dataset):
    def __init__(self, root, csv_file, image_field, target_field,
                 loader=default_loader, transform=None,
                 target_transform=None, add_extension=None,
                 limit=None, random_subset_size=None,
                 age_field = "age_approx", sex_field="sex",remove_unknowns=True,ctrl=False):
        """
        :param root:
        :param csv_file:
        :param image_field:
        :param target_field:
        :param loader:
        :param transform:
        :param target_transform:
        :param add_extension:
        :param limit:
        :param random_subset_size:
        :param age_field:
        :param sex_field:
        :param remove_unknowns: whether to remove unknowns or not
        :param ctrl: whether to return only healthy subjects (ctrl group) or not
        """
        self.root = root
        self.loader = loader
        self.image_field = image_field
        self.target_field = target_field
        self.transform = transform
        self.target_transform = target_transform
        self.add_extension = add_extension
        self.age_field = age_field
        self.sex_field = sex_field
        self.ctrl = ctrl
        self.data = pd.read_csv(csv_file)
        if remove_unknowns:
            self.data = self.data[self.data[self.sex_field] != "unknown"]
            self.data = self.data[self.data[self.age_field] != "unknown"]
            self.data.reset_index(inplace=True)

        if self.ctrl:
            self.data = self.data[self.data[self.target_field] == 0]
            self.data.reset_index(inplace=True)
            
        if random_subset_size:
            self.data = self.data.sample(n=random_subset_size)
            self.data = self.data.reset_index()

        if type(limit) == int:
            limit = (0, limit)
        if type(limit) == tuple:
            self.data = self.data[limit[0]:limit[1]]
            self.data = self.data.reset_index()

        classes = list(self.data[self.target_field].unique())
        classes.sort()
        self.class_to_idx = {classes[i]: i for i in range(len(classes))}
        self.classes = classes

        print('Found {} images from {} classes.'.format(len(self.data),
                                                       len(classes)))
        for class_name, idx in self.class_to_idx.items():
            n_images = dict(self.data[self.target_field].value_counts())
            print("    Class '{}' ({}): {} images.".format(
                class_name, idx, n_images[class_name]))

    def __getitem__(self, index):
        path = os.path.join(self.root,
                            self.data.loc[index, self.image_field])
        if self.add_extension:
            path = path + self.add_extension
        sample = self.loader(path)
        target = self.class_to_idx[self.data.loc[index, self.target_field]]
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.data)

class CSVDatasetWithName(CSVDataset):
    """
    CSVData that also returns image names.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        return super().__getitem__(i), name

class CSVDatasetWithMeta(CSVDataset):
    """
    CSVData that also returns image names, gender and age info.
    """

    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        name = self.data.loc[i, self.image_field]
        age = self.data.loc[i, self.age_field]
        gender = self.data.loc[i, self.sex_field]
        return super().__getitem__(i), name, age, gender

class CSVDatasetWithAllLabels(CSVDataset):
    """
    CSVData that also returns  and labels for gender and age.
    With the assumption that unknown data is removed. Otherwise they are assigned to the 3rd class.
    """
    def __getitem__(self, i):
        """
        Returns:
            tuple(tuple(PIL image, int), str): a tuple
            containing another tuple with an image and
            the label, and a string representing the
            name of the image.
        """
        age = int(self.data.loc[i, self.age_field])
        gender = self.data.loc[i, self.sex_field]
        if gender == "female":
            gender = 1
        elif gender == "male":
            gender = 0
        return super().__getitem__(i), age, gender