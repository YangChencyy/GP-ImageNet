import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import os
import random
from torchvision import transforms
import torchvision.datasets as datasets

np.random.seed(42)

class ImageNetSubset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

def load_data(data_folder):
    all_data = []
    all_labels = []
    for idx in range(1, 11):  # Assuming there are 10 batches for training
        batch_file = os.path.join(data_folder, f'train_data_batch_{idx}')
        with open(batch_file, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
            all_data.append(batch['data'])
            all_labels.extend(batch['labels'])
            # print(batch['labels'][0:20])
    return np.concatenate(all_data), np.array(all_labels)


def select_and_remap_classes(data, labels, num_classes=10):
    unique_classes = np.unique(labels)
    selected_classes = np.random.choice(unique_classes, num_classes, replace=False)
    
    # Create a mapping for the selected classes to [0, num_classes-1]
    class_mapping = {old_class: new_class for new_class, old_class in enumerate(selected_classes)}
    
    # Filter and remap labels
    selected_indices = [i for i, label in enumerate(labels) if label in selected_classes]
    remapped_labels = [class_mapping[label] for label in labels[selected_indices]]
    
    # Apply selection and remapping
    selected_data = data[selected_indices]
    
    return selected_data, np.array(remapped_labels), class_mapping


def load_test_data(data_folder, class_mapping):
    test_file = os.path.join(data_folder, 'val_data')  # Adjust the file name/path as necessary
    with open(test_file, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
    x = batch['data']
    y = batch['labels']

    # Adjust labels to match the training data's remapping and selection
    y = [class_mapping.get(label, -1) for label in y]  # Subtract 1 since original labels start at 1
    valid_indices = [i for i, label in enumerate(y) if label >= 0]
    x = x[valid_indices]
    y = [y[i] for i in valid_indices]

    return x, y


##################################  OOD Datasets   ############################################################

class INaturalistDataLoader:
    def __init__(self, root_dir, version='2021_train_mini', target_type='full', batch_size=32, download=True):
        self.root_dir = root_dir
        self.version = version
        self.target_type = target_type
        self.batch_size = batch_size
        self.download = download
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the INaturalist dataset
        dataset = datasets.INaturalist(root=self.root_dir, version=self.version, target_type=self.target_type, transform=self.transform, download=self.download)

        # Create and return a DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)




class SUN397DataLoader:
    def __init__(self, root_dir, batch_size=32, download=True):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.download = download
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = None  # Define if you need to apply any transformations to the labels

    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the SUN397 dataset
        dataset = datasets.SUN397(root=self.root_dir, transform=self.transform, target_transform=self.target_transform, download=self.download)

        # Create and return a DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)



class Places365DataLoader:
    def __init__(self, root_dir, split='val', small=True, download=True, batch_size=32):
        self.root_dir = root_dir
        self.split = split
        self.small = small
        self.download = download
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.target_transform = None  # Optional, define if needed

    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the Places365 dataset
        dataset = datasets.Places365(root=self.root_dir, split=self.split, small=self.small,
                                     download=self.download, transform=self.transform,
                                     target_transform=self.target_transform)

        # Create and return a DataLoader
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)


class DTDDataLoader:
    def __init__(self, root_dir, split='train', partition=1, download=False, batch_size=32):
        self.root_dir = root_dir
        self.split = split
        self.partition = partition
        self.download = download
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def get_data_loader(self, shuffle=True, num_workers=4):
        # Initialize the DTD dataset
        train_dataset = datasets.DTD(root=self.root_dir, split=self.split, partition=self.partition,
                               transform=self.transform, download=self.download)
        val_dataset = datasets.DTD(root=self.root_dir, split='val', transform=self.transform, download=True)
        combined_dataset = ConcatDataset([train_dataset, val_dataset])


        # Create and return a DataLoader
        print(len(combined_dataset))
        return DataLoader(combined_dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)

class SVHNDataLoader:
    def __init__(self, root_dir, split='train', download=True, batch_size=32):

        self.root_dir = root_dir
        self.split = split
        self.download = download
        self.batch_size = batch_size
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # Resize the image to 32x32 pixels
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
            # These normalization values are approximations; adjust as necessary.
        ])
        self.target_transform = None  # Optional, define if needed

    def get_data_loader(self, shuffle=True, num_workers=4):

        dataset = datasets.SVHN(root=self.root_dir, split=self.split, download=self.download,
                                transform=self.transform, target_transform=self.target_transform)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, num_workers=num_workers)
        return data_loader